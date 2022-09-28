########################
# Problem Set 1 | Coding
# Group Submission
# Members:
# 	Yasser Parambathkandy - (G01294910)
# 	Indranil Pal - (G01235186)
# Date: 09/01/2022
########################

def collatz_num(n):
    """
    Question 1: Write a function in Python which, when given any integer n, runs
    the game above until 1 is reached, printing out all the numbers visited along the way.
    """
    print('collatz processing', n)
    # (a) Pick a positive integer, call it n.
    if n == 1:  # avoid endless loop
        return 1
    else:
        n = odd_even_calc(n)
        return collatz_num(int(n))  # Repeat steps 2-4 with the new number.


def odd_even_calc(n):
    """
    odd even check in collatz algorithm. DRY.
    """
    if n % 2 == 0:  # (b) If n is even, divide it by 2.
        n = n / 2
    else:
        n = (3 * n) + 1  # (c) Otherwise, multiply it by 3 and add 1.
    return n


def collatz_steps(n, step=0):
    """
    Question 2: Write another function which returns (not prints) the number of steps it takes to go from n to 1.
    This should also be recursive.
    """
    if n == 1:  # avoid endless loop
        return step
    else:
        n = odd_even_calc(n)
        step += 1
        return collatz_steps(n, step)


def memoizer(function, result_dict):
    """
    Question 3: Write a function called memoizer which takes as its arguments a function, and a storage dictionary, and
    returns a new function, a memoized version of the function that was passed in. You can assume
    that the function-to-be-memoized takes only positional arguments
    """
    cache = result_dict  # cache collatz results

    def cacheable_function(n):  # wrapper that calls the original function
        if n in cache:
            return cache[n]
        result = function(n)  # n not yet calculated and not in cache
        cache[n] = result  # store in cache
        return result

    return cacheable_function  # memoized function


def max_collatz_steps(collatz_steps_memoized, n):
    """
    Question 4: Next up are two functions which will find for us values for n which take the longest to get back to 1.
    Write max collatz steps and max collatz num which return the largest number of steps, and the corresponding n
    out of all the numbers from 1 up to the passed in value.
    """
    steps = {}
    for num in range(1, n + 1):
        steps[num] = collatz_steps_memoized(num)  # store steps taken by each number in dict
    max_steps = max(steps.values())  # extract max steps and its corresponding number
    max_num = max(steps, key=steps.get)
    print('The maximum no. of steps taken was', max_steps, 'for the input', max_num)
    return steps


def max_collatz_num(collatz_num_memoized, n):
    """
    Question 4: Next up are two functions which will find for us values for n which take the longest to get back to 1.
    Write max collatz steps and max collatz num which return the largest number of steps, and the corresponding n
    out of all the numbers from 1 up to the passed in value.
    """
    for num in range(1, n + 1):
        collatz_num_memoized(num)


def collatz_num_chain(n, chain):
    """
    Question 5: Modify (or write a new function) which builds up a graph linking numbers to their Collatz successor".
    """
    if n == 1:
        return 1
    else:
        cur_val = n
        n = odd_even_calc(n)
        chain[cur_val] = int(n)  # The chain dict keeps track of value before and after thereby forming a chain
        return collatz_num_chain(int(n), chain)


def main():
    # question 3: use the memoizer to make a memoized version of your previous two functions.
    collatz_steps_memoized = memoizer(collatz_steps, {})
    collatz_num_memoized = memoizer(collatz_num, {})

    input_n = 10
    max_collatz_steps(collatz_steps_memoized, input_n)
    max_collatz_num(collatz_num_memoized, input_n)

    # Question 5: Modify (or write a new function) which builds up a graph linking numbers to their Collatz successor".
    collatz_chain = {}
    collatz_num_chain(input_n, collatz_chain)
    print('The collatz path chain', collatz_chain)


if __name__ == '__main__':
    main()
