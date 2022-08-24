def multiplicacion(a,b):
    print(f'la multiplicación de {a} X {b} es igual a {a*b}')
    print(f'Y este resultado es par: {par(a*b)}')

def par(a):
    if a % 2 ==0:
        return 'Par'
    else:
        return 'Impar'

print('inicia el software')
multiplicacion(3,5)
print("siguiente")
multiplicacion(6,8)
