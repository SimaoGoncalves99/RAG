import argparse
import markdown

def main(args):


    #TODO Implement main function
    print("Implement function")

    return




if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # Add arguments
    parser.add_argument("num1", type=int, help="The first number")
    parser.add_argument("num2", type=int, help="The second number")
    parser.add_argument("--verbose", action="store_true", help="Increase output verbosity")

    args = parser.parse_args()

    main(args)