// Solidity source for the public functions of a module, for the EVM.
//
// The EVM has no LLVM target. Its vendor language is Solidity, and the inline
// assembly of Solidity is Yul. So each function of the module becomes a Yul
// function. Each public function also becomes a Solidity function that calls
// its Yul function. A Solidity compiler makes the bytecode and the ABI from
// that file, as the Metal compiler does for src/dialect/MetalImage.cpp.
//
// A function arrives here as a CFG, with each local in a stack slot. Yul takes
// structured code. So this turns the stack slots of numbers into values and
// lifts the CFG to `scf.while` and `scf.if`. Then it prints Yul. An op that
// the printer does not know is an error that names the op.
//
// How a value sits in a 256-bit word of the EVM.
// - An integer of 8 to 128 bits has its sign bit in every bit above it, also
//   when its Vx type is unsigned. Solidity keeps an `intN` the same way, so a
//   signed parameter needs no conversion.
// - An `i1` is 0 or 1.
// - `index` is a 64-bit integer.
//
// The module attribute `vx.evm_exports` has one line for each public function,
// as `name(u64,i64)->u64`. It gives the Vx types of the source, because MLIR
// does not say whether an integer is signed. `void` is a function that returns
// nothing. The module attribute `vx.evm_contract` is the name of the contract.
//
// A failed `assert`, a division by zero and a signed division that overflows
// make the call revert with no data.

#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/ControlFlowToSCF/ControlFlowToSCF.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/IR/MemRefMemorySlot.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/CFGToSCF.h"
#include "mlir/Transforms/Mem2Reg.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"

#include <optional>
#include <string>

using namespace mlir;

namespace {

// The widest integer that the EVM holds in one word.
constexpr unsigned wordBits = 256;

// A Vx scalar type of a parameter or a result, as `vx.evm_exports` spells it.
struct Scalar {
  unsigned bits = 0;
  bool isSigned = false;
  bool isBool = false;
};

// A public function: its name, and the Vx types of its parameters and result.
struct Export {
  std::string name;
  SmallVector<Scalar> params;
  std::optional<Scalar> result;
};

// The scalar type that `name` spells, or nothing for a type that the EVM
// backend does not take.
std::optional<Scalar> scalarOf(StringRef name) {
  if (name == "bool")
    return Scalar{1, false, true};
  if (name == "usize")
    return Scalar{64, false, false};
  if (name.size() < 2 || (name[0] != 'i' && name[0] != 'u'))
    return std::nullopt;
  unsigned bits = 0;
  if (name.drop_front().getAsInteger(10, bits))
    return std::nullopt;
  if (bits < 8 || bits > 128 || !llvm::isPowerOf2_32(bits))
    return std::nullopt;
  return Scalar{bits, name[0] == 'i', false};
}

// Reads one line of `vx.evm_exports`. Returns false with `error` set.
bool parseExport(StringRef line, Export &out, std::string &error) {
  size_t open = line.find('(');
  size_t close = line.find(")->");
  if (open == StringRef::npos || close == StringRef::npos || close < open) {
    error =
        ("'" + line + "' in `vx.evm_exports` is not `name(types)->type`").str();
    return false;
  }
  out.name = line.take_front(open).str();
  SmallVector<StringRef> params;
  line.slice(open + 1, close).split(params, ',', -1, /*KeepEmpty=*/false);
  for (StringRef param : params) {
    std::optional<Scalar> scalar = scalarOf(param.trim());
    if (!scalar) {
      error = ("public function '" + out.name + "' has a parameter of type '" +
               param.trim() + "'. The EVM backend takes integers and `bool`")
                  .str();
      return false;
    }
    out.params.push_back(*scalar);
  }
  StringRef result = line.drop_front(close + 3).trim();
  if (result == "void")
    return true;
  out.result = scalarOf(result);
  if (!out.result) {
    error = ("public function '" + out.name + "' returns type '" + result +
             "'. The EVM backend returns an integer, a `bool` or nothing")
                .str();
    return false;
  }
  return true;
}

std::string solidityType(const Scalar &s) {
  if (s.isBool)
    return "bool";
  return ((s.isSigned ? "int" : "uint") + Twine(s.bits)).str();
}

// The number of bits of an integer type, with `index` as 64 bits. 0 for a type
// that the printer does not hold in a word.
unsigned bitsOf(Type t) {
  if (t.isIndex())
    return 64;
  auto integer = dyn_cast<IntegerType>(t);
  if (!integer)
    return 0;
  unsigned bits = integer.getWidth();
  if (bits == 1 || bits == wordBits)
    return bits;
  if (bits < 8 || bits > 128 || !llvm::isPowerOf2_32(bits))
    return 0;
  return bits;
}

// The Yul number with a 1 in each of the low `bits` bits.
std::string maskOf(unsigned bits) { return "0x" + std::string(bits / 4, 'f'); }

// `e` is a number whose low `bits` bits are right. Gives the same integer with
// its upper bits as the header of this file says.
std::string wrapped(unsigned bits, const Twine &e) {
  if (bits == 1)
    return ("and(" + e + ", 1)").str();
  if (bits == wordBits)
    return e.str();
  return ("signextend(" + Twine(bits / 8 - 1) + ", " + e + ")").str();
}

// The integer `e` of `bits` bits as an unsigned number: 0 in each upper bit.
std::string unsignedOf(unsigned bits, const Twine &e) {
  if (bits == 1 || bits == wordBits)
    return e.str();
  return ("and(" + e + ", " + maskOf(bits) + ")").str();
}

// The Yul number for the integer `value` of `bits` bits.
std::string literalOf(const APInt &value, unsigned bits) {
  if (bits == 1)
    return value.isZero() ? "0" : "1";
  APInt wide = value.sextOrTrunc(wordBits);
  if (wide.isNonNegative() && wide.getActiveBits() <= 64)
    return std::to_string(wide.getZExtValue());
  SmallString<80> digits;
  wide.toString(digits, 16, /*Signed=*/false, /*formatAsCLiteral=*/false,
                /*UpperCase=*/false);
  return ("0x" + digits).str();
}

// The name of the Yul function for the function `symbol`. Yul in Solidity
// takes no `.` and no `$` in a name that a program can also write.
std::string yulNameOf(StringRef symbol) {
  std::string out = "vx_";
  for (char c : symbol)
    out += (llvm::isAlnum(c) || c == '_') ? c : '_';
  return out;
}

// Whether a Solidity function can have the name of the public function
// `name`. The printer gives its own values names such as `x12`, and a Yul
// name cannot be the same as a Solidity name that it can see.
bool usableName(StringRef name) {
  static const char *const reserved[] = {
      "abstract", "address",   "anonymous", "as",       "assembly", "bool",
      "break",    "bytes",     "calldata",  "catch",    "constant", "continue",
      "contract", "delete",    "do",        "else",     "emit",     "enum",
      "error",    "event",     "external",  "fallback", "false",    "for",
      "function", "if",        "immutable", "import",   "indexed",  "interface",
      "internal", "is",        "library",   "mapping",  "memory",   "modifier",
      "new",      "override",  "payable",   "pragma",   "private",  "public",
      "pure",     "receive",   "return",    "returns",  "revert",   "storage",
      "string",   "struct",    "this",      "throw",    "true",     "try",
      "type",     "unchecked", "using",     "var",      "view",     "virtual",
      "while"};
  for (const char *word : reserved)
    if (name == word)
      return false;
  if (name.starts_with("vx_"))
    return false;
  for (StringRef stem : {"x", "y", "c", "d", "go", "a", "r"})
    if (name.starts_with(stem) && name.size() > stem.size() &&
        llvm::all_of(name.drop_front(stem.size()), llvm::isDigit))
      return false;
  return true;
}

// Prints the Yul function of one function at a time.
class Printer {
public:
  explicit Printer(ModuleOp module) : module(module), os(text) {}

  // The Yul function of `f`. Returns false with `error` set on failure.
  bool function(func::FuncOp f, std::string &out, std::string &error);

private:
  ModuleOp module;
  std::string text;
  llvm::raw_string_ostream os;
  std::string failure;
  int depth = 0;
  unsigned next = 0;
  DenseMap<Value, std::string> names;
  // The Yul names of the results of the function that is printed.
  SmallVector<std::string> returned;

  bool fail(Operation *op, const Twine &why) {
    if (failure.empty()) {
      std::string where;
      llvm::raw_string_ostream w(where);
      if (op)
        w << "'" << op->getName() << "': ";
      failure = where + why.str();
    }
    return false;
  }

  llvm::raw_ostream &line() {
    os.indent(4 * depth);
    return os;
  }

  std::string fresh(StringRef stem = "x") {
    return (stem + Twine(next++)).str();
  }

  std::string name(Value v) {
    auto it = names.find(v);
    assert(it != names.end() && "a value is used before the printer named it");
    return it->second;
  }

  std::string define(Value v) {
    std::string n = fresh();
    names[v] = n;
    return n;
  }

  // Declares a Yul variable for each of `values`. Fails for a value that does
  // not fit a word.
  bool declare(ValueRange values, Operation *op) {
    for (Value v : values) {
      if (!bitsOf(v.getType()))
        return fail(op, "a value of a type that the EVM backend does not hold");
      line() << "let " << define(v) << "\n";
    }
    return true;
  }

  void assign(ValueRange to, ValueRange from) {
    for (auto [t, f] : llvm::zip(to, from))
      line() << name(t) << " := " << name(f) << "\n";
  }

  bool block(Block &b);
  bool op(Operation *op);
  bool arith(Operation *op);
  bool cast(Operation *op);
  bool callOp(func::CallOp op);
  bool ifOp(scf::IfOp op);
  bool whileOp(scf::WhileOp op);
  bool switchOp(scf::IndexSwitchOp op);
};

bool Printer::function(func::FuncOp f, std::string &out, std::string &error) {
  text.clear();
  failure.clear();
  names.clear();
  returned.clear();
  next = 0;
  // Inside `contract`, `function` and `assembly`.
  depth = 3;

  auto refuse = [&](const Twine &why) {
    error = ("function '" + f.getSymName() + "': " + why).str();
    return false;
  };
  if (!f.getBody().hasOneBlock())
    return refuse("its branches could not be made structured");
  Block &entry = f.getBody().front();
  SmallVector<std::string> params;
  for (BlockArgument argument : entry.getArguments()) {
    if (!bitsOf(argument.getType()))
      return refuse(
          "a parameter has a type that the EVM backend does not hold");
    params.push_back(define(argument));
  }
  for (Type result : f.getFunctionType().getResults()) {
    if (!bitsOf(result))
      return refuse("a result has a type that the EVM backend does not hold");
    returned.push_back(fresh("y"));
  }

  line() << "function " << yulNameOf(f.getSymName()) << "("
         << llvm::join(params, ", ") << ")";
  if (!returned.empty())
    os << " -> " << llvm::join(returned, ", ");
  os << " {\n";
  ++depth;
  if (!block(entry))
    return refuse(failure);
  --depth;
  line() << "}\n";
  out = text;
  return true;
}

bool Printer::block(Block &b) {
  for (Operation &o : b)
    if (!op(&o))
      return false;
  return true;
}

bool Printer::op(Operation *o) {
  // The op that holds the region prints these.
  if (isa<scf::YieldOp, scf::ConditionOp>(o))
    return true;
  if (auto r = dyn_cast<func::ReturnOp>(o)) {
    for (auto [y, v] : llvm::zip(returned, r.getOperands()))
      line() << y << " := " << name(v) << "\n";
    return true;
  }
  if (auto i = dyn_cast<scf::IfOp>(o))
    return ifOp(i);
  if (auto w = dyn_cast<scf::WhileOp>(o))
    return whileOp(w);
  if (auto s = dyn_cast<scf::IndexSwitchOp>(o))
    return switchOp(s);
  if (auto a = dyn_cast<cf::AssertOp>(o)) {
    line() << "if iszero(" << name(a.getArg()) << ") { revert(0, 0) }\n";
    return true;
  }
  if (auto c = dyn_cast<func::CallOp>(o))
    return callOp(c);
  if (auto p = dyn_cast<ub::PoisonOp>(o)) {
    if (!bitsOf(p.getType()))
      return fail(o, "a value of a type that the EVM backend does not hold");
    line() << "let " << define(p.getResult()) << " := 0\n";
    return true;
  }
  if (o->getName().getDialectNamespace() == "arith")
    return arith(o);
  return fail(o, "the EVM backend does not print this operation yet");
}

bool Printer::callOp(func::CallOp c) {
  auto callee = module.lookupSymbol<func::FuncOp>(c.getCallee());
  if (!callee || callee.isExternal())
    return fail(c, "it calls '" + c.getCallee() +
                       "', a function that the EVM does not have");
  SmallVector<std::string> arguments;
  for (Value v : c.getOperands())
    arguments.push_back(name(v));
  std::string call =
      yulNameOf(c.getCallee()) + "(" + llvm::join(arguments, ", ") + ")";
  if (c.getNumResults() == 0) {
    line() << call << "\n";
    return true;
  }
  SmallVector<std::string> results;
  for (Value r : c.getResults()) {
    if (!bitsOf(r.getType()))
      return fail(c, "a result of a type that the EVM backend does not hold");
    results.push_back(define(r));
  }
  line() << "let " << llvm::join(results, ", ") << " := " << call << "\n";
  return true;
}

bool Printer::arith(Operation *o) {
  if (o->getNumResults() != 1 || !bitsOf(o->getResult(0).getType()))
    return fail(o, "a result of a type that the EVM backend does not hold");
  for (Value operand : o->getOperands())
    if (!bitsOf(operand.getType()))
      return fail(o, "an operand of a type that the EVM backend does not hold");
  Value result = o->getResult(0);
  unsigned bits = bitsOf(result.getType());

  if (auto c = dyn_cast<arith::ConstantOp>(o)) {
    auto integer = dyn_cast<IntegerAttr>(c.getValue());
    if (!integer)
      return fail(o, "a constant that is not an integer");
    line() << "let " << define(result)
           << " := " << literalOf(integer.getValue(), bits) << "\n";
    return true;
  }
  if (auto s = dyn_cast<arith::SelectOp>(o)) {
    std::string picked = define(result);
    line() << "let " << picked << " := " << name(s.getFalseValue()) << "\n";
    line() << "if " << name(s.getCondition()) << " { " << picked
           << " := " << name(s.getTrueValue()) << " }\n";
    return true;
  }
  if (auto c = dyn_cast<arith::CmpIOp>(o)) {
    unsigned width = bitsOf(c.getLhs().getType());
    std::string a = name(c.getLhs()), b = name(c.getRhs());
    std::string ua = unsignedOf(width, a), ub = unsignedOf(width, b);
    std::string e;
    switch (c.getPredicate()) {
    case arith::CmpIPredicate::eq:
      e = "eq(" + a + ", " + b + ")";
      break;
    case arith::CmpIPredicate::ne:
      e = "iszero(eq(" + a + ", " + b + "))";
      break;
    case arith::CmpIPredicate::slt:
      e = "slt(" + a + ", " + b + ")";
      break;
    case arith::CmpIPredicate::sle:
      e = "iszero(sgt(" + a + ", " + b + "))";
      break;
    case arith::CmpIPredicate::sgt:
      e = "sgt(" + a + ", " + b + ")";
      break;
    case arith::CmpIPredicate::sge:
      e = "iszero(slt(" + a + ", " + b + "))";
      break;
    case arith::CmpIPredicate::ult:
      e = "lt(" + ua + ", " + ub + ")";
      break;
    case arith::CmpIPredicate::ule:
      e = "iszero(gt(" + ua + ", " + ub + "))";
      break;
    case arith::CmpIPredicate::ugt:
      e = "gt(" + ua + ", " + ub + ")";
      break;
    case arith::CmpIPredicate::uge:
      e = "iszero(lt(" + ua + ", " + ub + "))";
      break;
    }
    // An `i1` is 0 or 1, not 0 or -1, so its order as a signed number is the
    // opposite of what MLIR says.
    bool isSigned = c.getPredicate() == arith::CmpIPredicate::slt ||
                    c.getPredicate() == arith::CmpIPredicate::sle ||
                    c.getPredicate() == arith::CmpIPredicate::sgt ||
                    c.getPredicate() == arith::CmpIPredicate::sge;
    if (width == 1 && isSigned)
      return fail(o, "a signed comparison of two `bool` values");
    line() << "let " << define(result) << " := " << e << "\n";
    return true;
  }
  if (o->getNumOperands() == 1)
    return cast(o);
  if (o->getNumOperands() != 2)
    return fail(o, "the EVM backend does not print this operation yet");

  std::string a = name(o->getOperand(0)), b = name(o->getOperand(1));
  std::string ua = unsignedOf(bits, a), ub = unsignedOf(bits, b);
  StringRef kind = o->getName().getStringRef();
  std::string e;
  if (kind == "arith.addi")
    e = wrapped(bits, "add(" + a + ", " + b + ")");
  else if (kind == "arith.subi")
    e = wrapped(bits, "sub(" + a + ", " + b + ")");
  else if (kind == "arith.muli")
    e = wrapped(bits, "mul(" + a + ", " + b + ")");
  else if (kind == "arith.divsi")
    e = wrapped(bits, "sdiv(" + a + ", " + b + ")");
  else if (kind == "arith.remsi")
    e = "smod(" + a + ", " + b + ")";
  else if (kind == "arith.divui")
    e = wrapped(bits, "div(" + ua + ", " + ub + ")");
  else if (kind == "arith.remui")
    e = wrapped(bits, "mod(" + ua + ", " + ub + ")");
  else if (kind == "arith.andi")
    e = "and(" + a + ", " + b + ")";
  else if (kind == "arith.ori")
    e = "or(" + a + ", " + b + ")";
  else if (kind == "arith.xori")
    e = "xor(" + a + ", " + b + ")";
  // Yul takes the number of bits first.
  else if (kind == "arith.shli")
    e = wrapped(bits, "shl(" + b + ", " + a + ")");
  else if (kind == "arith.shrsi")
    e = "sar(" + b + ", " + a + ")";
  else if (kind == "arith.shrui")
    e = wrapped(bits, "shr(" + b + ", " + ua + ")");
  else
    return fail(o, "the EVM backend does not print this operation yet");
  if (bits == 1 &&
      (kind == "arith.divsi" || kind == "arith.remsi" || kind == "arith.shrsi"))
    return fail(o, "a signed operation on `bool` values");
  line() << "let " << define(result) << " := " << e << "\n";
  return true;
}

// A conversion from one integer type to another.
bool Printer::cast(Operation *o) {
  Value result = o->getResult(0);
  Value from = o->getOperand(0);
  unsigned bits = bitsOf(result.getType());
  unsigned width = bitsOf(from.getType());
  std::string a = name(from);
  std::string e;
  if (isa<arith::ExtSIOp>(o) || (isa<arith::IndexCastOp>(o) && bits >= width)) {
    // A wider integer with the same sign bits is the same word. A `bool` that
    // is 1 becomes -1.
    e = width == 1 ? "sub(0, " + a + ")" : a;
  } else if (isa<arith::ExtUIOp>(o) ||
             (isa<arith::IndexCastUIOp>(o) && bits > width)) {
    e = unsignedOf(width, a);
  } else if (isa<arith::TruncIOp, arith::IndexCastOp, arith::IndexCastUIOp>(
                 o)) {
    e = wrapped(bits, a);
  } else {
    return fail(o, "the EVM backend does not print this operation yet");
  }
  line() << "let " << define(result) << " := " << e << "\n";
  return true;
}

// Yul has no `else`. So an `scf.if` with two branches is a `switch`.
bool Printer::ifOp(scf::IfOp i) {
  if (!declare(i.getResults(), i))
    return false;
  bool hasElse = !i.getElseRegion().empty() &&
                 (i.getNumResults() != 0 ||
                  !llvm::hasSingleElement(i.getElseRegion().front()));
  if (!hasElse) {
    line() << "if " << name(i.getCondition()) << " {\n";
    ++depth;
    if (!block(i.getThenRegion().front()))
      return false;
    --depth;
    line() << "}\n";
    return true;
  }
  line() << "switch " << name(i.getCondition()) << "\n";
  line() << "case 0 {\n";
  ++depth;
  if (!block(i.getElseRegion().front()))
    return false;
  assign(i.getResults(), i.elseYield().getOperands());
  --depth;
  line() << "}\n";
  line() << "default {\n";
  ++depth;
  if (!block(i.getThenRegion().front()))
    return false;
  assign(i.getResults(), i.thenYield().getOperands());
  --depth;
  line() << "}\n";
  return true;
}

// `scf.while` as a loop that never ends by itself: the before region decides,
// and the after region makes the values of the next turn.
bool Printer::whileOp(scf::WhileOp w) {
  SmallVector<std::string> carried, passed;
  if (!declare(w.getResults(), w))
    return false;
  line() << "{\n";
  ++depth;
  for (Value init : w.getInits()) {
    carried.push_back(fresh("c"));
    line() << "let " << carried.back() << " := " << name(init) << "\n";
  }
  line() << "for { } 1 { } {\n";
  ++depth;
  std::string go = fresh("go");
  line() << "let " << go << "\n";
  for (size_t i = 0; i < w.getNumResults(); ++i) {
    passed.push_back(fresh("d"));
    line() << "let " << passed.back() << "\n";
  }
  line() << "{\n";
  ++depth;
  Block &before = w.getBefore().front();
  for (auto [argument, c] : llvm::zip(before.getArguments(), carried)) {
    if (!bitsOf(argument.getType()))
      return fail(w, "a value of a type that the EVM backend does not hold");
    line() << "let " << define(argument) << " := " << c << "\n";
  }
  if (!block(before))
    return false;
  scf::ConditionOp condition = w.getConditionOp();
  line() << go << " := " << name(condition.getCondition()) << "\n";
  for (auto [d, v] : llvm::zip(passed, condition.getArgs()))
    line() << d << " := " << name(v) << "\n";
  --depth;
  line() << "}\n";
  line() << "if iszero(" << go << ") {\n";
  ++depth;
  for (auto [r, d] : llvm::zip(w.getResults(), passed))
    line() << name(r) << " := " << d << "\n";
  line() << "break\n";
  --depth;
  line() << "}\n";
  line() << "{\n";
  ++depth;
  Block &after = w.getAfter().front();
  for (auto [argument, d] : llvm::zip(after.getArguments(), passed)) {
    if (!bitsOf(argument.getType()))
      return fail(w, "a value of a type that the EVM backend does not hold");
    line() << "let " << define(argument) << " := " << d << "\n";
  }
  if (!block(after))
    return false;
  for (auto [c, v] : llvm::zip(carried, w.getYieldOp().getOperands()))
    line() << c << " := " << name(v) << "\n";
  --depth;
  line() << "}\n";
  --depth;
  line() << "}\n";
  --depth;
  line() << "}\n";
  return true;
}

bool Printer::switchOp(scf::IndexSwitchOp s) {
  if (!declare(s.getResults(), s))
    return false;
  line() << "switch " << name(s.getArg()) << "\n";
  auto body = [&](Region &region) {
    ++depth;
    bool ok = block(region.front());
    if (ok)
      assign(s.getResults(),
             llvm::cast<scf::YieldOp>(region.front().getTerminator())
                 .getOperands());
    --depth;
    line() << "}\n";
    return ok;
  };
  for (auto [value, region] : llvm::zip(s.getCases(), s.getCaseRegions())) {
    line() << "case " << literalOf(APInt(64, value, /*isSigned=*/true), 64)
           << " {\n";
    if (!body(region))
      return false;
  }
  line() << "default {\n";
  return body(s.getDefaultRegion());
}

// The functions that `f` calls, and those that they call, with `f` first.
void reachedFrom(ModuleOp module, func::FuncOp f,
                 SmallVectorImpl<func::FuncOp> &order,
                 llvm::StringSet<> &seen) {
  if (!seen.insert(f.getSymName()).second)
    return;
  order.push_back(f);
  f.walk([&](func::CallOp call) {
    auto callee = module.lookupSymbol<func::FuncOp>(call.getCallee());
    if (callee && !callee.isExternal())
      reachedFrom(module, callee, order, seen);
  });
}

// Makes each of `functions` structured: numbers in values instead of stack
// slots, and `scf` instead of a CFG.
LogicalResult structure(ModuleOp m, ArrayRef<func::FuncOp> functions,
                        std::string &error) {
  MLIRContext *context = m.getContext();
  {
    PassManager pm(context);
    pm.addPass(createLowerAffinePass());
    pm.addPass(createSCFToControlFlowPass());
    if (failed(pm.run(m))) {
      error = "making the loops of the module branches failed";
      return failure();
    }
  }
  for (func::FuncOp f : functions) {
    SmallVector<PromotableAllocationOpInterface> slots;
    f.walk([&](memref::AllocaOp a) {
      if (a.getType().getRank() == 0)
        if (auto p = dyn_cast<PromotableAllocationOpInterface>(*a))
          slots.push_back(p);
    });
    OpBuilder builder(context);
    DominanceInfo dominance(f);
    (void)tryToPromoteMemorySlots(slots, builder, DataLayout::closest(f),
                                  dominance);
  }
  {
    PassManager pm(context);
    pm.addPass(createCanonicalizerPass());
    if (failed(pm.run(m))) {
      error = "simplifying the module failed";
      return failure();
    }
  }
  ControlFlowToSCFTransformation lift;
  for (func::FuncOp f : functions) {
    DominanceInfo dominance(f);
    if (failed(transformCFGToSCF(f.getBody(), lift, dominance))) {
      error = "the branches of function '" + f.getSymName().str() +
              "' could not be made structured";
      return failure();
    }
  }
  {
    PassManager pm(context);
    pm.addPass(createCanonicalizerPass());
    pm.addPass(createCSEPass());
    if (failed(pm.run(m))) {
      error = "simplifying the structured module failed";
      return failure();
    }
  }
  return success();
}

// Whether the function `f` has the parameters and the result that `e` says.
bool sameSignature(func::FuncOp f, const Export &e) {
  FunctionType type = f.getFunctionType();
  if (type.getNumInputs() != e.params.size() ||
      type.getNumResults() != (e.result ? 1u : 0u))
    return false;
  for (auto [input, param] : llvm::zip(type.getInputs(), e.params))
    if (bitsOf(input) != param.bits)
      return false;
  return !e.result || bitsOf(type.getResult(0)) == e.result->bits;
}

} // namespace

namespace mlir {
namespace vx {

/// The Solidity source of the public functions of `module`, or the empty
/// string with `error` set. Works on a copy, so `module` does not change.
std::string evmImageOf(ModuleOp module, std::string &error) {
  auto exportsAttr = module->getAttrOfType<StringAttr>("vx.evm_exports");
  if (!exportsAttr) {
    error = "the module does not say which functions are public. The "
            "attribute `vx.evm_exports` is missing";
    return {};
  }
  StringRef contract = "VxContract";
  if (auto named = module->getAttrOfType<StringAttr>("vx.evm_contract"))
    contract = named.getValue();

  SmallVector<Export> exports;
  SmallVector<StringRef> lines;
  exportsAttr.getValue().split(lines, '\n', -1, /*KeepEmpty=*/false);
  for (StringRef line : lines) {
    Export e;
    if (!parseExport(line.trim(), e, error))
      return {};
    if (!usableName(e.name)) {
      error = "public function '" + e.name +
              "' has a name that the Solidity file cannot use. Give the "
              "function another name";
      return {};
    }
    exports.push_back(std::move(e));
  }
  if (exports.empty()) {
    error = "the program has no public function. Each `pub fn` becomes one "
            "function of the contract";
    return {};
  }

  OwningOpRef<ModuleOp> holder(llvm::cast<ModuleOp>(module->clone()));
  ModuleOp copy = holder.get();
  SmallVector<func::FuncOp> roots;
  SmallVector<func::FuncOp> reached;
  llvm::StringSet<> seen;
  for (const Export &e : exports) {
    auto f = copy.lookupSymbol<func::FuncOp>(e.name);
    if (!f || f.isExternal()) {
      error = "public function '" + e.name + "' has no code in the module";
      return {};
    }
    if (!sameSignature(f, e)) {
      error = "public function '" + e.name +
              "' has other parameters in the module than in `vx.evm_exports`";
      return {};
    }
    roots.push_back(f);
    reachedFrom(copy, f, reached, seen);
  }
  if (failed(structure(copy, reached, error)))
    return {};

  // Two functions whose names differ only in a character that Yul does not
  // take would get one Yul name.
  llvm::StringMap<std::string> yulNames;
  for (func::FuncOp f : reached) {
    auto [it, isNew] =
        yulNames.try_emplace(yulNameOf(f.getSymName()), f.getSymName().str());
    if (!isNew) {
      error = "functions '" + it->second + "' and '" + f.getSymName().str() +
              "' get the same Yul name";
      return {};
    }
  }

  Printer printer(copy);
  llvm::StringMap<std::string> bodies;
  for (func::FuncOp f : reached) {
    std::string body;
    if (!printer.function(f, body, error))
      return {};
    bodies[f.getSymName()] = std::move(body);
  }

  std::string image;
  llvm::raw_string_ostream os(image);
  os << "// SPDX-License-Identifier: UNLICENSED\n";
  os << "// vxc wrote this file. Change the Vx source, not this file.\n";
  os << "pragma solidity ^0.8.20;\n\n";
  os << "contract " << contract << " {\n";
  for (auto [index, e] : llvm::enumerate(exports)) {
    if (index != 0)
      os << "\n";
    SmallVector<std::string> params, arguments;
    for (auto [i, param] : llvm::enumerate(e.params)) {
      std::string name = ("a" + Twine(i)).str();
      params.push_back(solidityType(param) + " " + name);
      // An unsigned parameter comes with 0 in each upper bit.
      arguments.push_back(
          param.isSigned || param.isBool ? name : wrapped(param.bits, name));
    }
    os << "    function " << e.name << "(" << llvm::join(params, ", ")
       << ") external pure";
    if (e.result)
      os << " returns (" << solidityType(*e.result) << " r0)";
    os << " {\n";
    os << "        assembly (\"memory-safe\") {\n";
    std::string call =
        yulNameOf(e.name) + "(" + llvm::join(arguments, ", ") + ")";
    os.indent(12);
    if (!e.result)
      os << call << "\n";
    else if (e.result->isSigned || e.result->isBool)
      os << "r0 := " << call << "\n";
    else
      os << "r0 := " << unsignedOf(e.result->bits, call) << "\n";
    SmallVector<func::FuncOp> needed;
    llvm::StringSet<> visited;
    reachedFrom(copy, roots[index], needed, visited);
    for (func::FuncOp f : needed)
      os << bodies[f.getSymName()];
    os << "        }\n";
    os << "    }\n";
  }
  os << "}\n";
  return image;
}

} // namespace vx
} // namespace mlir

namespace {

struct ExportEvmPass
    : public PassWrapper<ExportEvmPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ExportEvmPass)

  llvm::StringRef getArgument() const override { return "vx-export-evm"; }

  llvm::StringRef getDescription() const override {
    return "Writes the public functions of the module as a Solidity contract "
           "for the EVM";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<scf::SCFDialect, ub::UBDialect, arith::ArithDialect>();
    memref::registerMemorySlotExternalModels(registry);
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    std::string error;
    std::string image = mlir::vx::evmImageOf(module, error);
    if (image.empty()) {
      module.emitError() << error;
      signalPassFailure();
      return;
    }
    module->setAttr("vx.exported_evm", StringAttr::get(&getContext(), image));
  }
};

} // namespace

namespace mlir {
namespace vx {

std::unique_ptr<Pass> createExportEvmPass() {
  return std::make_unique<ExportEvmPass>();
}

} // namespace vx
} // namespace mlir
