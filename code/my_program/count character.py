def tm000():

	s = "xcvdsdddfdsvdsvdvd"
	max_time = 0 
	result ={}
	for i in s:
		if s.count(i)> max_time:
			result.clear()
			result[i] =s.count(i)
			max_time = s.count(i)
		elif s.count(i) == max_time:
			result[i] = s.count(i)
	print (result)
		# print (i)

''''
题目001：有四个数字：1、2、3、4，能组成多少个互不相同且无重复数字的三位数？各是多少？
'''
def tm001_1():
	arr = []
	for i in range(1,5):
		for j in range(1,5):
			for k in range(1,5):
				num =100*i+10*j+k
				if i!=j and i!=k and j!=k and num not in arr:
					arr.append(num)
	print(len(arr),arr)				
'''
题目002：企业发放的奖金根据利润(I)的多少来提成：
低于或等于10万元时，奖金可提10%；
利润高于10万元，低于20万元时，低于10万元的部分按10%提成，高于10万元的部分，可提成7.5%；
20万到40万之间时，高于20万元的部分，可提成5%；
40万到60万之间时高于40万元的部分，可提成3%；
60万到100万之间时，高于60万元的部分，可提成1.5%；
高于100万元时，超过100万元的部分按1%提成。
从键盘输入当月利润I，求应发放奖金总数？
'''
def tm002():
	money = int(input('salary:'))
	arr = [1000000, 600000, 400000, 200000, 100000,0]
	rat = [0.01, 0.015,0.03,0.05,0.075,0.01]
	for i in range(len(arr)):
		if money >arr[i]:
			bonus += (money - arr[i]*rat[i])
			money = arr[i]
	print(bonus)
'''
题目003：一个整数，它加上100后是一个完全平方数，
再加上168又是一个完全平方数，请问该数是多少？
'''
def tm003():
	import math
	for i in range(1000):
		x = math.sqrt(i+100)
		y = math.sqrt(i+168)
		if x%1 ==0 and y%1 ==0:
			print(i)
'''
题目004：输入某年某月某日，判断这一天是这一年的第几天？
'''
def tm004():
    '''
    【个人备注】：知道python有时间元组这一概念，这道题完全不需要计算。
    时间元组包含九个属性
    tm_year     年
    tm_mon      月(1~12)
    tm_mday     日(1~31)
    tm_hour     时(0~23)
    tm_min      分(0~59)
    tm_sec      秒(0~61, 60或61是闰秒)
    tm_wday     星期(0~6, 0是周一)
    tm_yday     第几天(1~366, 366是儒略历)
    tm_isdst    夏令时(平时用不到)
    '''
    import time
    date = input('输入时间(例如2018-01-23):')
    st = time.strptime(date,'%Y-%m-%d') # 时间文本转化成时间元祖
    num = st.tm_yday
    print(num)
tm004()
