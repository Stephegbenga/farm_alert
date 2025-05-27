from gpiozero import LED

from time import sleep


led = LED(17)


def make_sound():
    led.on()
    sleep(2)
    led.off()



