
import pickle

nameOfSavefile=input("Write name of file to write to> ")

if nameOfSavefile == "":
    nameOfSavefile = "test_of_saving"

def save_object(obj, filename):
    with open(filename, 'wb') as output:  # Overwrites any existing file.
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)


def load(filename):
    with open(filename, 'rb') as input:
        return pickle.load(input)


# sample usage
save_object(kalle, nameOfSavefile)

print(load(nameOfSavefile))
