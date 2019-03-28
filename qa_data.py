import xlrd
import codecs
data = xlrd.open_workbook('knowledge/qa.xlsx')
sheet2 = data.sheet_by_index(0)
cols0 = sheet2.col_values(0)
cols1 = sheet2.col_values(1)
cols2 = sheet2.col_values(2)
n=0
line_seen=set()
with codecs.open('qa_data.txt','w',encoding='utf8') as f1, codecs.open('my_data.txt','w',encoding='utf8') as f2:

    for i in range(sheet2.nrows):
        if cols1[i] != '':
            if cols0[i] not in line_seen:
                line_seen.add(cols0[i])
                n+=1
                f1.write(cols0[i]+'|&|'+cols1[i])
                f1.write('\n')

                f2.write('text,intent,'+'slot'+str(n))
                f2.write('\n')
                a2=cols2[i]
                f2.write(cols0[i] + '|intent' + str(n) + '|' + a2.replace(' ','，'))
                f2.write('\n')
        else:
            if cols0[i] not in line_seen:
                line_seen.add(cols0[i])
                a2=cols2[i]
                f2.write(cols0[i] + '|intent' + str(n) + '|' + a2.replace(' ','，'))
                f2.write('\n')