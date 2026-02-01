"""
Simple Calculator Application
Supports basic arithmetic operations: +, -, *, /, **, %
"""

class Calculator:
    """A simple calculator class for performing arithmetic operations."""
    
    def __init__(self):
        self.last_result = 0
    
    def add(self, a, b):
        """Add two numbers."""
        self.last_result = a + b
        return self.last_result
    
    def subtract(self, a, b):
        """Subtract b from a."""
        self.last_result = a - b
        return self.last_result
    
    def multiply(self, a, b):
        """Multiply two numbers."""
        self.last_result = a * b
        return self.last_result
    
    def divide(self, a, b):
        """Divide a by b."""
        if b == 0:
            raise ValueError("Cannot divide by zero")
        self.last_result = a / b
        return self.last_result
    
    def power(self, a, b):
        """Raise a to the power of b."""
        self.last_result = a ** b
        return self.last_result
    
    def modulo(self, a, b):
        """Get remainder of a divided by b."""
        if b == 0:
            raise ValueError("Cannot divide by zero")
        self.last_result = a % b
        return self.last_result
    
    def get_last_result(self):
        """Get the last calculated result."""
        return self.last_result


def evaluate_expression(expression):
    """
    Safely evaluate a mathematical expression with support for parentheses.
    Only allows numbers and operators: +, -, *, /, **, %
    """
    # Remove spaces
    expression = expression.replace(" ", "")
    
    # Validate input - only allow numbers, operators, and parentheses
    allowed_chars = set("0123456789+-*/%().")
    if not all(c in allowed_chars for c in expression):
        raise ValueError("Invalid characters in expression")
    
    # Use eval with a restricted namespace for safety
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return result
    except ZeroDivisionError:
        raise ValueError("Cannot divide by zero")
    except SyntaxError:
        raise ValueError("Invalid expression syntax")
    except Exception as e:
        raise ValueError(f"Invalid expression: {e}")


def main():
    """Main function to run the calculator in interactive mode."""
    calc = Calculator()
    print("=" * 50)
    print("Welcome to the Calculator!")
    print("=" * 50)
    print("Operations: + (add), - (subtract), * (multiply)")
    print("           / (divide), ** (power), % (modulo)")
    print("Parentheses: ( ) for grouping expressions")
    print("Type 'quit' to exit\n")
    
    while True:
        try:
            user_input = input("Enter calculation (e.g., (5 + 3) * 2): ").strip()
            
            if user_input.lower() == 'quit':
                print("Thank you for using the calculator!")
                break
            
            if not user_input:
                continue
            
            # Evaluate the expression
            result = evaluate_expression(user_input)
            calc.last_result = result
            print(f"Result: {user_input} = {result}\n")
        
        except ValueError as e:
            print(f"Error: {e}\n")
        except Exception as e:
            print(f"Error: Invalid input or operation. {e}\n")


if __name__ == "__main__":
    main()
