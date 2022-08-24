'''
el ciclo for posee la siguiente sintaxis:
for elemento in iterable:
    codigo
'''
"""
numeros=[8,7,8,2,6,1,9,25,45]

for n in numeros:
    print(n)    

#imprime el termino del diccionario
valores={'A':3,'E':4,'I':5,'O':4,'R':8}

for d in valores:
    print(d)


#imprime el valor del diccionario
valores={'A':3,'E':4,'I':5,'O':4,'R':8}

for h in valores.values():
    print(h)


#imprime el valor y termino del diccionario
valores={'A':3,'E':4,'I':5,'O':4,'R':8}

for d, v in valores.items():
    print('d= ',d, ' ,v= ',v)


#clase range

for i in range(11):
    print(i)
"""
#clase range incremento range(min, max-1, paso)

for i in range(0,11,2):
    print(i)
"""
"""
