# Error Handling Guide for OCDocker

## Overview

OCDocker uses a standardized error handling system based on the `Error` class. This guide explains when and how to use the Error class system versus standard Python exceptions.

## Error Class System

The `Error` class provides a consistent way to report errors throughout the codebase. It uses error codes and report levels to provide structured error reporting.

### Key Components

1. **ErrorCode Enum**: Defines all possible error codes (e.g., `FILE_NOT_EXIST`, `VALUE_ERROR`, `DOCKING_FAILED`)
2. **ReportLevel Enum**: Defines severity levels (`NONE`, `ERROR`, `WARNING`, `INFO`, `SUCCESS`, `DEBUG`)
3. **Error Class**: Provides static methods for each error code (e.g., `Error.file_not_exist()`)

### Import Pattern

```python
import OCDocker.Error as ocerror
from OCDocker.Error import Error, ErrorCode, ReportLevel
```

## When to Use Error Class vs Standard Exceptions

### Use Error Class for: ✅

1. **User-facing errors** - Errors that can occur during normal operation:
   - File operations (file not found, file exists, read/write errors)
   - Configuration errors (missing settings, invalid values)
   - Validation errors (invalid input from user)
   - Business logic errors (docking failed, molecule not prepared)

2. **Recoverable errors** - Errors that the caller can handle:
   - Missing optional resources
   - Validation failures that can be retried
   - Warnings about non-critical issues

3. **Errors that should be logged** - Errors that need consistent logging:
   - Errors that need to appear in logs with timestamps
   - Errors that need severity levels (WARNING vs ERROR)

### Use Standard Exceptions for: ✅

1. **Programming errors** - Bugs in the code itself:
   - Invalid function arguments (use `TypeError`, `ValueError` for parameter validation)
   - Internal state corruption (use `RuntimeError`)
   - Assertion failures (use `AssertionError`)

2. **System errors** - Errors that indicate system-level problems:
   - Out of memory (use `MemoryError`)
   - Import errors (use `ImportError`)
   - Attribute errors for missing attributes (use `AttributeError`)

3. **Third-party library errors** - When wrapping external libraries:
   - SQLAlchemy errors (use `SQLAlchemyError`)
   - JSON parsing errors (use `JSONDecodeError`)

## Usage Patterns

### Pattern 1: Return Error Code (Recommended for User-facing Errors)

```python
def load_molecule(path: str) -> Optional[Molecule]:
    """Load a molecule from a file.
    
    Returns
    -------
    Optional[Molecule]
        The loaded molecule, or None if an error occurred.
    """
    if not os.path.exists(path):
        Error.file_not_exist(f"File not found: {path}")
        return None
    
    try:
        # Load molecule...
        return molecule
    except Exception as e:
        Error.parse_molecule(f"Failed to parse molecule: {e}")
        return None

# Caller checks for None
molecule = load_molecule("ligand.pdb")
if molecule is None:
    # Error already reported by Error class
    return ErrorCode.FILE_NOT_EXIST
```

### Pattern 2: Return Error Code from Function

```python
def prepare_receptor(receptor_path: str) -> int:
    """Prepare a receptor for docking.
    
    Returns
    -------
    int
        ErrorCode.OK if successful, otherwise an error code.
    """
    if not os.path.exists(receptor_path):
        return Error.file_not_exist(f"Receptor file not found: {receptor_path}")
    
    # ... preparation logic ...
    
    if not preparation_successful:
        return Error.receptor_not_prepared("Receptor preparation failed")
    
    return ErrorCode.OK

# Caller checks error code
result = prepare_receptor("receptor.pdbqt")
if result != ErrorCode.OK:
    return result  # Propagate error code
```

### Pattern 3: Raise Standard Exception (For Programming Errors)

```python
def calculate_descriptor(molecule: Molecule, descriptor_type: str) -> float:
    """Calculate a molecular descriptor.
    
    Parameters
    ----------
    molecule : Molecule
        The molecule to calculate descriptor for.
    descriptor_type : str
        Type of descriptor to calculate.
    
    Returns
    -------
    float
        The descriptor value.
    
    Raises
    ------
    TypeError
        If molecule is not a Molecule instance.
    ValueError
        If descriptor_type is not a valid option.
    """
    # Parameter validation - raise standard exceptions
    if not isinstance(molecule, Molecule):
        raise TypeError(f"Expected Molecule instance, got {type(molecule)}")
    
    if descriptor_type not in VALID_DESCRIPTORS:
        raise ValueError(f"Invalid descriptor type: {descriptor_type}")
    
    # ... calculation logic ...
    return value
```

### Pattern 4: Hybrid Approach (Validate Inputs, Use Error for Operations)

```python
def run_docking(ligand_path: str, receptor_path: str, box_file: str) -> int:
    """Run docking simulation.
    
    Parameters
    ----------
    ligand_path : str
        Path to ligand file.
    receptor_path : str
        Path to receptor file.
    box_file : str
        Path to box definition file.
    
    Returns
    -------
    int
        ErrorCode.OK if successful, otherwise an error code.
    
    Raises
    ------
    TypeError
        If any parameter is not a string.
    """
    # Parameter validation - raise standard exceptions
    if not isinstance(ligand_path, str):
        raise TypeError("ligand_path must be a string")
    
    # Operational errors - use Error class
    if not os.path.exists(ligand_path):
        return Error.file_not_exist(f"Ligand file not found: {ligand_path}")
    
    if not os.path.exists(receptor_path):
        return Error.file_not_exist(f"Receptor file not found: {receptor_path}")
    
    # ... docking logic ...
    
    if docking_failed:
        return Error.docking_failed("Docking simulation failed")
    
    return ErrorCode.OK
```

## Common Error Code Mappings

When converting from standard exceptions to Error class:

| Standard Exception | Error Code | Usage |
|-------------------|------------|-------|
| `FileNotFoundError` | `Error.file_not_exist()` | User-provided file doesn't exist |
| `PermissionError` | `Error.file_not_exist()` or custom | File access denied |
| `ValueError` (user input) | `Error.value_error()` | Invalid user-provided value |
| `ValueError` (internal) | `raise ValueError()` | Programming error in parameters |
| `KeyError` (user data) | `Error.data_not_found()` | Missing required data in user input |
| `KeyError` (internal) | `raise KeyError()` | Programming error in data structure |
| `TypeError` (user input) | `Error.wrong_type()` | User provided wrong type |
| `TypeError` (internal) | `raise TypeError()` | Programming error in function call |

## Best Practices

1. **Be Consistent**: Within a module, use the same pattern consistently
2. **Document Exceptions**: If a function can raise exceptions, document them in docstring
3. **Propagate Error Codes**: When calling functions that return error codes, propagate them
4. **Use Appropriate Levels**: Choose the right `ReportLevel` (ERROR vs WARNING)
5. **Provide Context**: Always include a descriptive message
6. **Don't Mix Patterns**: Don't raise exceptions and return error codes in the same function

## Examples

### Good: Error Class for User-facing Errors

```python
def load_config(config_path: str) -> Optional[Dict]:
    """Load configuration from file."""
    if not os.path.exists(config_path):
        Error.file_not_exist(f"Config file not found: {config_path}")
        return None
    
    try:
        with open(config_path) as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        Error.read_file(f"Invalid JSON in config file: {e}")
        return None
```

### Good: Standard Exceptions for Programming Errors

```python
def normalize_vector(v: np.ndarray) -> np.ndarray:
    """Normalize a vector."""
    if not isinstance(v, np.ndarray):
        raise TypeError("v must be a numpy array")
    
    if v.ndim != 1:
        raise ValueError("v must be 1-dimensional")
    
    norm = np.linalg.norm(v)
    if norm == 0:
        raise ValueError("Cannot normalize zero vector")
    
    return v / norm
```

### Bad: Mixing Patterns Inconsistently

```python
def bad_example(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError("File not found")  # Should use Error.file_not_exist()
    
    if isinstance(path, int):
        return Error.wrong_type("Path is int")  # Should raise TypeError
    
    # Inconsistent: mixing exceptions and error codes
```

## Migration Strategy

When refactoring code to use consistent error handling:

1. **Identify user-facing vs programming errors**: 
   - User-facing → Error class
   - Programming errors → Standard exceptions

2. **Update function signatures**: 
   - Add return type hints (e.g., `-> int` for error codes)
   - Document exceptions in docstrings

3. **Update callers**: 
   - Check for `None` or error codes instead of catching exceptions
   - Handle error codes appropriately

4. **Test thoroughly**: 
   - Ensure error messages are still clear
   - Verify error codes are propagated correctly

## Summary

- **User-facing errors** → Use `Error` class with error codes
- **Programming errors** → Use standard Python exceptions
- **Be consistent** → Use the same pattern within a module
- **Document everything** → Document return codes and exceptions
- **Provide context** → Always include descriptive messages

