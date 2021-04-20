#Runs the code for Task1

print("Run Feature Extract")
exec(open('./feature_extract_Alex.py').read())
print("Run Fine Tune")
exec(open('./fine_tune_Alex.py').read())

# Runs the code for task 2

print("train on MNIST")
exec(open('./MNIST.py').read())
print("MNIST test on SVHN")
exec(open('./SVHN.py').read())
print("MNIST feature_extract on SVHN")
exec(open('./MNISTonSVHN.py').read())
