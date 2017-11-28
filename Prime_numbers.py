


def return_prime(list_of_numbers):
	d = []
	for a in n:
		for b in range(1,a+1):
			c = a/b

			if c.is_integer() == True:
				d.append(c)


			if len(d) > 2:
				print (a, ' is non prime')
				break

		if len(d) <= 2:
			print (a,' is prime')

		d = []


n = [7 , 8 , 9, 11, 13 , 1823 ]
return_prime(n)

