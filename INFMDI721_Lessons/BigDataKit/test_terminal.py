from sys import argv

def hello_world(n):
	for i in range(n):
		print('Hello World !')

if __name__ == '__main__':
	n = int(argv[1])
	hello_world(n)	 
