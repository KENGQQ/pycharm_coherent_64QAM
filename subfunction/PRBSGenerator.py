
def PRBSGenerator(prbsnum, length=100000):
    if prbsnum == 13:
        prbs13 = []
        init = [1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0]  # initinal seed
        for indx in range(length):
            XOR = init[0] ^ init[2] ^ init[3] ^ init[12]      #XOR cell
            init.insert(0, XOR)
            init.pop()
            prbs13.append(XOR)

        with open('PRBS_13.txt', 'w') as outfile:
            outfile.write("\n".join(str(item) for item in prbs13))

    if prbsnum == 15:
        prbs15 = []
        init = [1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1]  # initinal seed
        for indx in range(length):
            XOR = init[13] ^ init[14]     #XOR cell
            init.insert(0, XOR)
            init.pop()
            prbs15.append(XOR)
        with open('PRBS_15.txt', 'w') as outfile:
            outfile.write("\n".join(str(item) for item in prbs15))

# PRBSGenerator(13)