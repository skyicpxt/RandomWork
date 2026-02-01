"""Simple number-guessing game."""
# this is a small game to guess a number.
import random

MAX_GUESSES = 3

def main():
    low, high = 1, 20
    secret = random.randint(low, high)
    guesses = 0

    print(f"I'm thinking of a number between {low} and {high}. You have {MAX_GUESSES} tries. Can you guess it?\n")

    while True:
        try:
            guess = int(input("Your guess: "))
        except ValueError:
            print("Please enter a number.\n")
            continue

        guesses += 1

        if guess < secret:
            if guesses >= MAX_GUESSES:
                print(f"Game over! You used all {MAX_GUESSES} tries. The number was {secret}.")
                break
            print("Too low! Try again.\n")
        elif guess > secret:
            if guesses >= MAX_GUESSES:
                print(f"Game over! You used all {MAX_GUESSES} tries. The number was {secret}.")
                break
            print("Too high! Try again.\n")
        else:
            print(f"Correct! You got it in {guesses} guess{'es' if guesses != 1 else ''}.")
            break

if __name__ == "__main__":
    main()
