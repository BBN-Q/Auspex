"""Plain-plan normalization and correctness checks, independent of authoring.

Normalization may rewrite a detached document. Validation never rewrites it.
Instruction IDs are assigned only after validation, in a separate final pass.
"""
from dataclasses import dataclass, field
import json
import math


class CompileError(ValueError):
    """An experiment cannot be represented as a valid executable plan."""


PLAN_VERSION = 4  # Plan versions are not silently upgraded during WIP development.


def plain(value):
    """Detach JSON data, normalizing NumPy values even inside nested containers."""
    def normalize(item):
        if hasattr(item, "tolist"):
            item = item.tolist()
        if isinstance(item, dict):
            return {key: normalize(child) for key, child in item.items()}
        if isinstance(item, (list, tuple)):
            return [normalize(child) for child in item]
        return item

    return json.loads(json.dumps(normalize(value), allow_nan=False))


def check_type(value, kind):
    """Check an already normalized scalar; do not coerce floats to integers."""
    accepted_types = {
        "bool": (bool,),
        "int": (int,),
        "float": (int, float),
        "str": (str,),
    }
    if type(value) not in accepted_types.get(kind, ()):
        raise CompileError(f"Expected finite {kind}, got {value!r}")
    if type(value) is float and not math.isfinite(value):
        raise CompileError(f"Expected finite {kind}, got {value!r}")


def check_parameter(name, spec, value):
    check_type(value, spec["type"])
    bounds = spec["bounds"]
    if bounds is not None and not bounds[0] <= value <= bounds[1]:
        raise CompileError(f"Parameter {name} outside bounds")


CHILD_BLOCKS = ("body", "then", "otherwise")
PHASES = ("setup", "main", "teardown")
NUMERIC_TYPES = {"int", "float"}
ARITHMETIC = {"add", "sub", "mul", "div"}
COMPARISONS = {"lt", "le", "gt", "ge", "eq", "ne"}
BOOLEAN_OPERATORS = {"all", "any", "not"}


def walk_instructions(nodes):
    for node in nodes:
        yield node
        for block in CHILD_BLOCKS:
            yield from walk_instructions(node.get(block, []))


def swept_parameters(document):
    names = set()
    for phase in PHASES:
        for node in walk_instructions(document[phase]):
            if node["op"] == "SWEEP":
                names.update(node["parameters"])
    return names


def updated_parameters(document):
    """Parameters explicitly assigned by the runtime program, including cleanup."""
    return {node["name"] for phase in PHASES
            for node in walk_instructions(document[phase]) if node["op"] == "SET_PARAMETER"}


def is_frozen(expr, swept):
    if expr["op"] == "literal":
        return True
    return expr["op"] == "parameter" and expr["name"] not in swept


def frozen_value(expr, parameters, swept):
    if expr["op"] == "literal":
        return expr["value"]
    if expr["op"] == "parameter" and expr["name"] not in swept:
        return parameters[expr["name"]]["value"]
    raise CompileError("Expected a literal or frozen unswept parameter")


def normalize_plan(document):
    """Resolve frozen loop bounds, leaving runtime expressions symbolic."""
    dynamic = swept_parameters(document) | updated_parameters(document)
    parameters = document["parameters"]
    for phase in PHASES:
        for node in walk_instructions(document[phase]):
            if node["op"] not in ("WHILE", "REPEAT"):
                continue
            limit = node["max_iterations"]
            if isinstance(limit, dict):
                node["max_iterations"] = frozen_value(limit, parameters, dynamic)
            if node["op"] == "REPEAT" and is_frozen(node["count"], dynamic):
                count = frozen_value(node["count"], parameters, dynamic)
                node["count"] = {"op": "literal", "value": count}


def assign_instruction_ids(document):
    """Stable structural paths; no semantic validation or normalization here."""
    def assign(nodes, path):
        for index, node in enumerate(nodes):
            node["id"] = f"{path}.{index:03d}"
            for block in CHILD_BLOCKS:
                suffix = "else" if block == "otherwise" else block
                assign(node.get(block, []), node["id"] + "." + suffix)

    for phase in PHASES:
        assign(document[phase], phase)


def explain_value(value):
    """Render plan values without making users read expression-tree JSON."""
    if isinstance(value, list):
        return "[" + ", ".join(explain_value(item) for item in value) + "]"
    if not isinstance(value, dict):
        return json.dumps(value)

    op = value.get("op")
    if op == "literal":
        return json.dumps(value["value"])
    if op == "parameter":
        return "$" + value["name"]
    if op == "ref":
        return "%" + value["name"]
    if op == "field":
        return explain_value(value["value"]) + "[" + json.dumps(value["name"]) + "]"
    symbols = {"lt": "<", "le": "<=", "gt": ">", "ge": ">=", "eq": "==",
               "ne": "!=", "add": "+", "sub": "-", "mul": "*", "div": "/"}
    if op in symbols:
        left, right = value["args"]
        return f"({explain_value(left)} {symbols[op]} {explain_value(right)})"
    if op == "neg":
        return "(-" + explain_value(value["args"][0]) + ")"
    if op in BOOLEAN_OPERATORS:
        arguments = ", ".join(explain_value(arg) for arg in value["args"])
        return f"{op}({arguments})"
    fields = [f"{key}: {explain_value(item)}" for key, item in sorted(value.items())]
    return "{" + ", ".join(fields) + "}"


@dataclass
class FlowState:
    """Values definitely available at an instruction, not runtime values."""
    references: dict = field(default_factory=dict)
    variables: set = field(default_factory=set)
    parameters: set = field(default_factory=set)

    def copy(self):
        return FlowState(dict(self.references), set(self.variables), set(self.parameters))

    @classmethod
    def intersection(cls, left, right):
        references = {name: kind for name, kind in left.references.items()
                      if right.references.get(name) == kind}
        return cls(references, left.variables & right.variables,
                   left.parameters & right.parameters)


class PlanValidator:
    """Validate known types and definite availability without executing a plan."""
    def __init__(self, document):
        self.document = document
        self.parameters = document["parameters"]
        self.swept = swept_parameters(document)
        self.updated = updated_parameters(document)
        self.dynamic = self.swept | self.updated

    def validate(self):
        if self.document["version"] != PLAN_VERSION:
            raise CompileError(f"Unsupported experiment plan version: expected {PLAN_VERSION}")
        expected = {"version", "pipelines", "instruments", "parameters", "setup", "main",
                    "teardown", "outputs", "metadata"}
        if set(self.document) != expected:
            raise CompileError(f"Incompatible experiment plan fields: {sorted(set(self.document) ^ expected)}")
        from ._experiment_publication import validate_attachments
        from ..filters.core import PipelineError
        try:
            self.connected_outputs = validate_attachments(self.document)
        except (PipelineError, ValueError, KeyError, TypeError) as exc:
            raise CompileError(f"Pipeline: {exc}") from exc
        main = self.document["main"]
        if (len(main) < 2 or main[0]["op"] != "SCAN_BEGIN" or main[-1]["op"] != "SCAN_END"
                or main[0]["name"] != main[-1]["name"]):
            raise CompileError("Publication plans require one explicit main scan envelope")
        boundaries = [node for phase in ("setup", "main", "teardown")
                      for node in walk_instructions(self.document[phase])
                      if node["op"] in ("SCAN_BEGIN", "SCAN_END")]
        if len(boundaries) != 2:
            raise CompileError("Nested or additional scan boundaries are not supported")
        from ..instruments.kinds import instrument_kind
        for role, spec in self.document["instruments"].items():
            if not role or not all(isinstance(spec[key], str) and spec[key] for key in ("kind", "name")):
                raise CompileError("Invalid instrument binding")
            try:
                instrument_kind(spec["kind"])
            except ValueError as exc:
                raise CompileError(str(exc)) from exc

        initial = FlowState()
        for name, spec in self.parameters.items():
            if name in self.dynamic and spec["value"] is None:
                continue
            check_parameter(name, spec, spec["value"])
            initial.parameters.add(name)

        after_setup = self.validate_body(self.document["setup"], initial, "setup")
        self.validate_body(self.document["main"], after_setup, "main")
        # Cleanup can run after any setup/main failure, even before the first point.
        self.validate_body(self.document["teardown"], initial, "teardown", teardown=True)

    def expression_type(self, expr, state):
        op = expr["op"]
        if op == "literal":
            return type(expr["value"]).__name__
        if op == "parameter":
            name = expr["name"]
            if name not in self.parameters:
                raise CompileError(f"Unknown parameter: {name}")
            if name not in state.parameters:
                raise CompileError(f"Parameter {name} is not initialized at this point")
            return self.parameters[name]["type"]
        if op == "ref":
            if expr["name"] not in state.references:
                raise CompileError(f"Value not defined on every path: {expr['name']}")
            return state.references[expr["name"]]
        if op == "field":
            if type(expr["name"]) not in (str, int):
                raise CompileError("Result selection requires a field name or integer index")
            self.expression_type(expr["value"], state)
            return "unknown"
        if op == "neg":
            if len(expr["args"]) != 1:
                raise CompileError("Negation requires one operand")
            kind = self.expression_type(expr["args"][0], state)
            if kind not in NUMERIC_TYPES | {"unknown"}:
                raise CompileError("Numeric operand required for negation")
            return kind
        if op not in ARITHMETIC | COMPARISONS | BOOLEAN_OPERATORS:
            raise CompileError(f"Unknown expression: {op}")

        types = [self.expression_type(arg, state) for arg in expr["args"]]
        if op in ARITHMETIC | COMPARISONS and len(types) != 2:
            raise CompileError("Binary expression requires two operands")
        if op == "not" and len(types) != 1:
            raise CompileError("NOT requires one operand")
        if op in BOOLEAN_OPERATORS:
            if any(kind not in ("bool", "unknown") for kind in types):
                raise CompileError("Boolean expression required")
            return "bool"
        if op in COMPARISONS:
            self.check_comparison(op, types)
            return "bool"
        if any(kind not in NUMERIC_TYPES | {"unknown"} for kind in types):
            raise CompileError("Numeric operands required")
        if "unknown" in types:
            return "unknown"
        if op == "div" or "float" in types:
            return "float"
        return "int"

    @staticmethod
    def check_comparison(op, types):
        if "unknown" in types:
            return  # The runtime result is not known to this compiler slice.
        left, right = types
        if left in NUMERIC_TYPES and right in NUMERIC_TYPES:
            return
        if left == right == "str":
            return
        if op in ("eq", "ne") and (left == right or "NoneType" in types):
            return
        raise CompileError(f"Incompatible comparison operands: {left} {op} {right}")

    def check_expression(self, expr, state, allowed=None):
        kind = self.expression_type(expr, state)
        if allowed is not None and kind not in allowed and kind != "unknown":
            raise CompileError(f"Expected {allowed}, got {kind}")
        return kind

    def check_assignment(self, expr, state, kind):
        allowed = ("float", "int") if kind == "float" else (kind,)
        self.check_expression(expr, state, allowed)

    def validate_body(self, nodes, incoming, path, teardown=False):
        state = incoming.copy()
        for index, node in enumerate(nodes):
            location = f"{path}.{index:03d}"
            try:
                state = self.validate_instruction(node, state, location, teardown)
            except CompileError as exc:
                raise CompileError(f"{location}: {exc}") from exc
        return state

    def validate_instruction(self, node, state, path, teardown):
        op = node["op"]
        if op in ("GET", "SET", "CALL"):
            self.validate_command(node, state, teardown)
        elif op == "PUBLISH":
            if not path.startswith("main.") or teardown:
                raise CompileError("Publication is allowed only inside the main scan")
            name = node["output"]
            if name not in self.connected_outputs:
                raise CompileError(f"Published output {name} has no pipeline connection")
            schema = self.document["outputs"][name]
            if set(node["axes"]) != set(schema["axis_names"]):
                raise CompileError(f"Publication axes disagree with output {name}")
            self.check_expression(node["data"], state)
            for expr in node["axes"].values():
                self.check_expression(expr, state)
        elif op == "WAIT":
            duration = node["duration"]
            self.check_expression(duration, state, ("int", "float"))
            if duration["op"] == "parameter" and duration["name"] in self.updated:
                # Its current value is checked for finiteness/nonnegativity by WAIT.
                # A SET_PARAMETER may also have validated stricter declared bounds.
                if duration["name"] in self.swept:
                    raise CompileError("WAIT parameter must not be swept")
            else:
                value = frozen_value(duration, self.parameters, self.swept)
                check_type(value, "float")
                if value < 0:
                    raise CompileError("Negative WAIT")
        elif op == "SET_PARAMETER":
            name = node["name"]
            if name not in self.parameters:
                raise CompileError(f"Unknown parameter: {name}")
            if name in self.swept:
                raise CompileError("Runtime updates to swept parameters are not supported yet")
            spec = self.parameters[name]
            self.check_assignment(node["value"], state, spec["type"])
            if node["value"]["op"] == "literal":
                check_parameter(name, spec, node["value"]["value"])
            state.parameters.add(name)
        elif op == "DECLARE":
            if node["name"] in state.references:
                raise CompileError("Duplicate variable")
            if node["type"] not in ("int", "float", "bool", "str"):
                raise CompileError("Unsupported variable type")
            self.check_assignment(node["value"], state, node["type"])
            state.variables.add(node["name"])
            state.references[node["name"]] = node["type"]
        elif op == "ASSIGN":
            if node["name"] not in state.variables:
                raise CompileError("Assignment to undeclared variable")
            self.check_assignment(node["value"], state, state.references[node["name"]])
        elif op == "IF":
            self.check_expression(node["condition"], state, ("bool",))
            after_then = self.validate_body(node["then"], state, path + ".then", teardown)
            after_else = self.validate_body(node["otherwise"], state, path + ".else", teardown)
            state = FlowState.intersection(after_then, after_else)
        elif op in ("WHILE", "REPEAT", "SWEEP"):
            if teardown:
                raise CompileError("Loops are not allowed in teardown in this slice")
            if op == "SWEEP":
                state = self.validate_sweep(node, state, path)
            else:
                self.validate_loop(node, state, path)
        elif op == "TERMINATE":
            if teardown or node["outcome"] not in ("SUCCEEDED", "FAILED"):
                raise CompileError("Invalid termination")
        elif op not in ("SCAN_BEGIN", "SCAN_END"):
            raise CompileError(f"Unknown instruction: {op}")
        return state

    def validate_command(self, node, state, teardown):
        op, role, member = node["op"], node["instrument"], node["member"]
        if role not in self.document["instruments"] or not isinstance(member, str):
            raise CompileError("Unknown instrument or invalid member")
        capabilities = self.document["instruments"][role].get("capabilities")
        if capabilities is not None:
            if op in ("GET", "SET"):
                spec = capabilities["settings"].get(member)
                permission = "read" if op == "GET" else "write"
                if spec is None or not spec[permission]:
                    raise CompileError(f"Undeclared or disallowed {op}: {member}")
                if set(node["kwargs"]) != set(spec["args"]):
                    raise CompileError(f"Incorrect setting arguments: {member}")
            elif member not in capabilities["operations"]:
                raise CompileError(f"Undeclared operation: {member}")
        for expr in node["kwargs"].values():
            self.check_expression(expr, state)
        for expr in node.get("args", []):
            self.check_expression(expr, state)
        if op == "SET":
            self.check_expression(node["value"], state)
        state.references[node["result"]] = "unknown"

    def validate_loop(self, node, state, path):
        limit = node["max_iterations"]
        if limit is not None and (type(limit) is not int or limit < 1):
            raise CompileError("Invalid loop bound")
        if node["op"] == "WHILE":
            self.check_expression(node["condition"], state, ("bool",))
            if limit is None:
                raise CompileError("WHILE requires positive max_iterations")
        else:
            self.check_expression(node["count"], state, ("int",))
            if node["count"]["op"] == "literal":
                count = node["count"]["value"]
                check_type(count, "int")
                if count < 0:
                    raise CompileError("Negative repetition count")
                if limit is not None and count > limit:
                    raise CompileError("Repeat count exceeds maximum")
            elif limit is None:
                raise CompileError("Dynamic loops require positive max_iterations")
        # General loops can execute zero times. Their declarations do not escape.
        self.validate_body(node["body"], state, path + ".body")

    def validate_sweep(self, node, state, path):
        names = node["parameters"]
        if not node["points"] or not names or any(name not in self.parameters for name in names):
            raise CompileError("Invalid sweep")
        for point in node["points"]:
            values = [point] if len(names) == 1 else point
            if not isinstance(values, list) or len(values) != len(names):
                raise CompileError("Joint sweep point width mismatch")
            for name, value in zip(names, values):
                check_parameter(name, self.parameters[name], value)
        loop_state = state.copy()
        loop_state.parameters.update(names)
        after_body = self.validate_body(node["body"], loop_state, path + ".body")
        # Fixed sweeps are nonempty: a reachable end hook has visited a point.
        # Keep result/variable scoping conservative, as for other loops.
        state.parameters.update(after_body.parameters)
        return state
