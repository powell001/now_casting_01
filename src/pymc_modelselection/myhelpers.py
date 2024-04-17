from colorama import Fore, Back, Style, init
init(autoreset=True)

def printme(data):
    print(Fore.RED + "DataFrame type: \n", data.dtypes)
    print(Fore.RED + "DataFrame head: \n", data.head())
    print(Fore.RED + "DataFrame tail: \n", data.tail())
    print(Fore.RED + "DataFrame shape: \n", data.shape)
    print(Fore.RED + "DataFrame describe: \n", data.describe())
    print(Fore.RED + "DataFrame index type: \n", data.index.dtype)
    data.info(verbose = True)
