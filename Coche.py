class Coche:
    # atributos
    ruedas=4
    color=''
    aceleracion=0
    velocidad=0

    #constructor
    def __init__(self, color, aceleracion):
        self.color=color
        self.aceleracion=aceleracion
        self.velocidad=0
    #metodos
    def acelerar(self):
        self.velocidad =self.velocidad+self.aceleracion
        return self.velocidad

class autoVolador(Coche):
    ruedas=6
    #constructor
    def __init__(self, color, aceleracion,esta_volando=False):
        super().__init__(color, aceleracion)
        self.esta_volando=esta_volando
        
    #metodo
    def vuela(self):
        self.esta_volando=True
        return "estoy volando"

c1=Coche('rojo',20)
print(c1.color, " ", c1.acelerar())


cv1=autoVolador('negro',60)
print(cv1.color)
print(cv1.vuela())

c2=Coche('azul',20)
print(c2.color)
