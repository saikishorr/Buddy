import sqlite3

conn = sqlite3.connect("chat_history.db")
cursor = conn.cursor()

cursor.execute("SELECT * FROM chat_logs ORDER BY timestamp DESC")
rows = cursor.fetchall()

for row in rows:
    print(f"ID: {row[0]} | Question: {row[1]} | Answer: {row[2]} | Time: {row[3]}")

conn.close()
