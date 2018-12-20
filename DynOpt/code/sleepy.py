'''
Created on Dec 13, 2018

@author: ameier
'''

if __name__ == '__main__':
    import time
    minute = 0
    while True:
        minute += 1
        print("minute: ", minute, flush=True)
        time.sleep(60)  # Delay for 1 minute (60 seconds).
