from datetime import datetime


def date_time_print(*output, sep=' ', end=''):
    if any(output):
        print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), end=' | ')
        for i in output[:-1]:
            print(i, end=sep)
        if any(end):
            print(output[-1], end=end)
        else:
            print(output[-1])
    else:
        print()
