def makeRegistrar():
     registry = {}
     def registrar(func):
         registry[func.__name__] = func
         return func  # normally a decorator returns a wrapped function, 
                      # but here we return func unmodified, after registering it
     registrar.all = registry
     return registrar

#Shape features
SHAPE = makeRegistrar()

#File features
FILE = makeRegistrar()

#Features that require multiplication
MULT = makeRegistrar()

ALL = [SHAPE, FILE, MULT]

#This is a bit sad, but basically this is a trick to make sure that all annotated features
#are detected by modules that just import annotations.py. All feature files will need to be
#added to this list to be recognized.
#import file_features
#import shape_features