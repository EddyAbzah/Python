import smtplib

sender = "McDonalds.Herzliya@McDonalds.com"
receivers = ['Tal.Gelkop@solaredge.com']

message = """From: McDonald's Herzliya <Ovads.Sabich@Herzelia.com>
To: Tal Gelkop <Tal.Gelkop@solaredge.com>
Subject: הודעה חשובה למזמיני תן ביס
שלום TAL Gelkop
עקב חשש לחיידקי ליסטריה במטבח,
אנחנו מעניקים לך קוד הנחה על סה"כ ההזמנה שבוצעה היום.
אנו חוששים כי מנות הנאגטס מסניפנו בהרצליה היו מודבקים
במידה ואתה מרגיש היטב, המשך כרגיל
לבעיות הרגש חופשי לפנות אל מנהל הסניף
מנהל סניף הרצליה: עומר זכריה
+972 54-768-6399
"""

message = """From: McDonald's Herzliya <Ovads.Sabich@Herzelia.com>
To: Tal Gelkop <Tal.Gelkop@solaredge.com>
Subject: הודעה חשובה למזמיני תן ביס
מתנצילם שוב על הטרחה.
למימוש ההטבה, הזן באתר תן ביס:
IDODEBIHASNOMANA
"""

message = message.encode("utf-8")
try:
    smtpObj = smtplib.SMTP('cust59304-s.out.mailcontrol.com')
    smtpObj.sendmail(sender, receivers, message)
    print("Successfully sent email")
except:
    print("Error: unable to send email")
