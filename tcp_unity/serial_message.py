import time

import serial

ser = serial.Serial(port='COM6',baudrate=9600,timeout=1)
ser.bytesize=serial.EIGHTBITS
ser.stopbits=serial.STOPBITS_ONE
ser.parity=serial.PARITY_NONE

command="brakelight:0,activate;"
ser.write(command.encode())
time.sleep(1)
command='brakelight:0,close;'
ser.write(command.encode())
# ser.close()


