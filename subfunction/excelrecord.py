from openpyxl import Workbook
from openpyxl.utils import get_column_letter
import openpyxl
import datetime
import os


def open_excel(address):
    if not os.path.exists(address + 'Record.xlsx'):
        print('creat an excel file')
        wb = Workbook()
        ws = wb.create_sheet('trial_1')
        del wb['Sheet']

    else:
        wb = openpyxl.load_workbook(address + 'Record.xlsx')
        ws = wb.create_sheet('trial_' + str(len(wb.worksheets) + 1), 0)

    ws['A1'] = datetime.datetime.now()

    parameter = ['eyepos', 'CMA[mean, type, overhead, taps, \nstepsize, iterator, earlystop, step_adj]', 'PLL BW',
                 'V-V[tap,r1,r2]', 'SNR,EVM,BERcount', 'Vol[type,tap]', 'SNR,EVM']

    ws.column_dimensions['A'].width = 20.0
    ws.column_dimensions['B'].width = 8.0
    ws.column_dimensions['C'].width = 38.0
    ws.column_dimensions['D'].width = 10.0
    ws.column_dimensions['E'].width = 15.0
    ws.column_dimensions['F'].width = 20.0
    ws.column_dimensions['G'].width = 20.0
    ws.column_dimensions['H'].width = 20.0
    ws.row_dimensions[1].height = 30

    for i in range(len(parameter)):
        ws.cell(row=1, column=i + 2, value=parameter[i])

    wb.save(address + 'Record.xlsx')


def write_excel(address, parameter_record):
    wb = openpyxl.load_workbook(address + 'Record.xlsx')
    ws = wb.worksheets[0]
    maxrow = ws.max_row
    for i in range(len(parameter_record)):
        ws.cell(row=maxrow + 1, column=1, value=datetime.datetime.now())
        ws.cell(row=maxrow + 1, column=i + 2, value=parameter_record[i])

    wb.save(address + 'Record.xlsx')
