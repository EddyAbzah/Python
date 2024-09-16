import smtplib

sender = ["McDonald's Herzliya", "McDonalds.Herzliya@McDonalds.com"]
receivers = ["Noam Dahan", ['Noam.Dahan@solaredge.com']]

message = f"""From: {sender[0]} <{sender[1]}>
To: {receivers[0]} <{receivers[1][0]}>
Subject: בדיקה בדיקה יא שרמוטה
שלום TAL Gelkop
עקב חשש לחיידקי ליסטריה במטבח,
אנחנו מעניקים לך קוד הנחה על סה"כ ההזמנה שבוצעה היום.
אנו חוששים כי מנות הנאגטס מסניפנו בהרצליה היו מודבקים
במידה ואתה מרגיש היטב, המשך כרגיל
לבעיות הרגש חופשי לפנות אל מנהל הסניף
מנהל סניף הרצליה: עומר זכריה
+972 54-768-6399
"""

message = message.encode("utf-8")
try:
    smtpObj = smtplib.SMTP('mail.solaredge.com')
    smtpObj.sendmail(sender[1], receivers[1], message)
    print("Successfully sent email")
except:
    print("Error: unable to send email")
