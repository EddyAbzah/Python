import xlsxwriter


#This is the main script which creates the
def runScript(LOGpath,fileSavePath):
    print(fileSavePath)
    print(LOGpath)
    KAstring='KaSignalLast = '
    TargetVoRef='TargetVoRef =  '
    VOstring='V0 = '
    BarkerSnrdBstring= 'BarkerSnrdB = '
    PacketSnrdBstring = 'PacketSnrdB = '
    MaxKaSignal=0
    MinKaSignal=100
    AverageKA=0
    MaxPacketSNR=0
    MinPacketSNR=100
    AveragePacketSNR=0
    MaxBarkerSNR=0
    MinBarkerSNR=100
    AverageBarkerSNR=0
    newList=[]
    nonFlag=False
    workbook = xlsxwriter.Workbook(fileSavePath+'.xlsx')
    worksheet1 = workbook.add_worksheet('Simple')
    with open(str(LOGpath)) as f:
        lines = f.readlines()
    for i in lines:
        testingString=i.split()
        if(len(testingString)>1):
            dateString=testingString[0]+' '+testingString[1]
            if KAstring in i:
                index=i.find(KAstring)+len(KAstring)
                newKAString=i[index:index+8]
            else:
                nonFlag=True
            if TargetVoRef in i:
                index=i.find(TargetVoRef)+len(TargetVoRef)
                refString=i[index:index+8]
                if not refString[3].isdigit():
                    refString=i[index:index+1]
            else:
                nonFlag=True
            if VOstring in i:
                index=i.find(VOstring)+len(VOstring)
                newVOstring=i[index:index+8]
                for word in newVOstring:
                    if not (word.isdigit() or word=='-' or word=='.'):
                        newVOstring=newVOstring[0]
            else:
                nonFlag = True
            if BarkerSnrdBstring in i:
                index = i.find(BarkerSnrdBstring) + len(BarkerSnrdBstring)
                newBarkerSnrdBstring=i[index:index+9]
            else:
                nonFlag=True
            if PacketSnrdBstring in i:
                index = i.find(PacketSnrdBstring) + len(PacketSnrdBstring)
                newPacketSnrdBstring=i[index:index+8]
            else:
                nonFlag=True
            if nonFlag==False:
                new_row_data=[dateString,float(newKAString),float(refString),float(newVOstring),float(newBarkerSnrdBstring),float(newPacketSnrdBstring)]
                newList.append(new_row_data)
                if float(newKAString)<MinKaSignal:
                    MinKaSignal=float(newKAString)
                elif float(newKAString)>MaxKaSignal:
                    MaxKaSignal=float(newKAString)
                if float(newBarkerSnrdBstring)<MinBarkerSNR:
                    MinBarkerSNR=float(newBarkerSnrdBstring)
                elif float(newBarkerSnrdBstring)>MaxBarkerSNR:
                    MaxBarkerSNR=float(newBarkerSnrdBstring)
                if float(newPacketSnrdBstring)<MinPacketSNR:
                    MinPacketSNR=float(newPacketSnrdBstring)
                elif float(newPacketSnrdBstring)>MaxPacketSNR:
                    MaxPacketSNR=float(newPacketSnrdBstring)
                AverageKA+=float(newKAString)
                AverageBarkerSNR+=float(newBarkerSnrdBstring)
                AveragePacketSNR+=float(newPacketSnrdBstring)
            nonFlag=False
    AverageKA=AverageKA/len(newList)
    AveragePacketSNR=AveragePacketSNR/len(newList)
    AverageBarkerSNR=AverageBarkerSNR/len(newList)
    lineIndex=len(newList)+1
    lineIndex='G'+str(lineIndex)
    worksheet1.add_table('B1:'+lineIndex,{'columns':[{'header':'Date'},{'header':'Ka Amplitude'},{'header':'Target Vout Reff'},{'header':'Vo of string'},{'header':'Barker SNR'},{'header':'Packet SNR'}]})
    worksheet1.set_column('A:G',23)
    for i in range(len(newList)):
        worksheet1.write_row(i+1, 1, newList[i])

    #Chart for KA last
    chart = workbook.add_chart({'type': 'line'})
    chart.add_series({'values': '=Simple!$C$2:$C$'+str(len(newList)+1),'name':'Ka Amplitude'})
    worksheet1.insert_chart('K2',chart)

    #Chart for Barker SNR
    chart = workbook.add_chart({'type': 'line'})
    chart.add_series({'values': '=Simple!$F$2:$F$'+str(len(newList)+1),'name':'Barker SNR'})
    worksheet1.insert_chart('K18',chart)

    #Chart for Packet SNR
    chart = workbook.add_chart({'type': 'line'})
    chart.add_series({'values': '=Simple!$G$2:$G$'+str(len(newList)+1),'name':'Packet SNR'})
    worksheet1.insert_chart('K34',chart)

    #Create Average Max Min Table
    worksheet1.add_table('B'+str(len(newList)+5)+':'+'E'+str(len(newList)+8),{'columns':[{'header':'Param Name'},{'header':'Average'},{'header':'Max'},{'header':'Min'}]})
    worksheet1.write_row(len(newList)+5,1,['Ka Amplitude',AverageKA,MaxKaSignal,MinKaSignal])
    worksheet1.write_row(len(newList)+6,1,['Barker SNR',AverageBarkerSNR,MaxBarkerSNR,MinBarkerSNR])
    worksheet1.write_row(len(newList)+7,1,['Packet SNR',AveragePacketSNR,MaxPacketSNR,MinPacketSNR])
    workbook.close()

