import pandas as pd
import numpy as np
completeSpam = pd.read_csv("completeSpamAssassin.csv")
# 删除空行
df = pd.DataFrame(completeSpam)
df.dropna(subset=['Body'], inplace=True)
# 拆分每个邮件为单词列表
df['words'] = df['Body'].apply(lambda email: 
 [word for line in email.split('\n') for word in line.split()])