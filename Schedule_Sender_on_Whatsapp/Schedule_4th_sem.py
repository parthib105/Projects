import pywhatkit as wp
import time
import numpy as np

class subject:
    def __init__(self, n, p, d):
        self.name = n
        self.place = p
        self.day = d
    

class exam:
    def __init__(self, n, p, dt, dy, t):
        self.name = n
        self.place = p
        self.date = dt
        self.day = dy
        self.time = t
        
        
def schedule(subjects):
    da = time.ctime(time.time())[0:2]
    # da = "Mo"
    msg = "Today's class timings\n\n"
    # msg = ""
    
    temp_arr = []
    for i in subjects:
        for j in i.day:
            if (j == da):
                temp_arr.append(subject(i.name, i.place, i.day[j])) #i.day[j] is timings of class on specific day
                
    # sorting according to time
    for i in range(len(temp_arr) - 1):
        for j in range(i+1, len(temp_arr)):
            if (int(temp_arr[j].day[0:2]) < int(temp_arr[i].day[0:2])):
                temp_arr[j], temp_arr[i] = temp_arr[i], temp_arr[j]
    
    # creating string of message
    for i in temp_arr:
            
        if (int(i.day[0:2]) > 12):
            t = f"0{int(i.day[:2]) - 12}{i.day[2:12]}0{int(i.day[12:14]) - 12}{i.day[14:]}"
            msg += "Course :  " + i.name + "\nVenue  :  " + i.place + "\nTime   :  " + t + "\n\n"
        elif (int(i.day[:2]) < 12 and int(i.day[12:14]) > 12):
            t = f"{i.day[:12]}0{int(i.day[12:14]) - 12}{i.day[14:]}"
            msg += "Course :  " + i.name + "\nVenue  :  " + i.place + "\nTime   :  " + t + "\n\n"
        else:
            msg += "Course :  " + i.name + "\nVenue  :  " + i.place + "\nTime   :  " + i.day + "\n\n"
        
    return msg 

# function to convert date into days (1 to 365)
def date_to_day(date : str ) -> int :        # date = dd/mm/yyyy
    t = time.ctime()
    months = [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    
    if (int(t[-4:]) % 4 == 0):
        months[2] = 29
        
    m = int(date[3:5])
    left_sum = np.sum(months[:m])
    return left_sum + int(date[:2])

# sorting function according to time
def srt(exams):
    n = len(exams)
    exm = exams
    for i in range(n - 1):
        for j in range(i + 1, n):
            if (int(exm[i].date[:2]) < int(exm[j].date[:2])):
                exm[j], exm[i] = exm[i], exm[j]
    return exm


def exams_schedule(exams : np.ndarray):
    n = len(exams)
    
    # sorting according to date of exam
    # for i in range(n - 1):
    #     for j in range(i+1, n):
    #         days_i = date_to_day(exams.date)
    #         days_j = date_to_day(exams.date)
    #         if (days_j < days_i):
    #             exams[j], exams[i] = exams[i], exams[j]
    
    exam_schd = []
    # sort according to time
    i = 0
    j = 1
    while (i < n):
        if (j < n and exams[i].date == exams[j].date):
            j += 1
        else:
            if (j == n - 1):
                arr = srt(exams[i:j+1])
            else:
                arr = srt(exams[i:j])
            exam_schd.append(arr)
            i = j
            j +=1 
            
    msg = "Today's exam timings\n\n"
    for i in exam_schd:
        msg += f"*{i[0].date} , {i[0].day}*\n"
        for j in i:
            # 09:00 AM to 09:55 AM
            if (int(j.time[:2]) > 12):
                t = f"0{int(j.time[:2]) - 12}{j.time[2:12]}0{int(j.time[12:14]) - 12}{j.time[14:]}"
                msg += "Course :  " + j.name + "\nVenue  :  " + j.place + "\nTime   :  " + t + "\n\n"
            elif (int(j.time[:2]) < 12 and int(j.time[12:14]) > 12):
                t = f"{j.time[:12]}0{int(j.time[12:14]) - 12}{j.time[14:]}"
                msg += "Course :  " + j.name + "\nVenue  :  " + j.place + "\nTime   :  " + t + "\n\n"
            else:
                msg += "Course :  " + j.name + "\nVenue  :  " + j.place + "\nTime   :  " + j.time + "\n\n"
        msg += "\n"
    
    return msg
    


if (__name__ == "__main__"):
    
    ############################## DAILY SCHEDULES ##############################
    
    subjects = []
    
    # 1-2 segment
    
    # subjects.append(subject("Introduction to Statistics", "Research Centre Complex", {"Mo" : "09:00 to 09:55", "We" : "11:00 to 11:55", "Th" : "10:00 to 10:55"}))
    
    # 3-4 segment
    
    # subjects.append(subject("Complex Variables", "Auditorium", {"Tu" : "16:00 to 17:25", "Fr" : "14:30 to 15:55"}))
    
    # 5-6 segment
    
    subjects.append(subject("Foundation in Machine Learning", "A-LH1", {"Mo" : "11:00 AM to 11:55 AM", "We" : "10:00 AM to 10:55 AM", "Th" : "09:00 AM to 09:55 AM"}))
    subjects.append(subject("Operating Systems II ", "A-LH2", {"Mo" : "16:00 PM to 17:25 PM", "We" : "16:00 PM to 18:55 PM","Th" : "14:30 PM to 15:55 PM"}))
    subjects.append(subject("Computational Methods in Material Science", "MS-LH3", {"Tu" : "14:30 PM to 15:55 PM", "Fr" : "16:00 PM to 17:25 PM"}))
    subjects.append(subject("Heat and Mass Transfer", "C-LH3", {"Mo" : "10:00 AM to 10:55 AM", "We" : "09:00 AM to 09:55 AM", "Th" : "11:00 AM to 11:55 AM"}))
    subjects.append(subject("Molecular and Cellular Biology", "BT-009", {"Mo" : "14:30 PM to 15:55 PM", "Th" : "16:00 PM to 17:25 PM"}))
    subjects.append(subject("Computer Aided Numerical Methods II", "C-306", {"Tu" : "10:00 AM to 11:25 AM", "Fr" : "09:30 AM to 10:55 AM"}))    

    # custom message
    # "Course : Molecular and Cellular Biology (exam)\nVenue  : BT-009\nTime   : 12:00 to 01:00\n"
    
    # print(schedule(subjects))
    # wp.sendwhatmsg_to_group_instantly("CrrCGsFykPnCm912n4S0yp", display(arr)) 
    
    
    ############################## EXAMS ##############################
    
    exams = []
    # exams.append(exam("Operating Systems II (quiz)", "A-LH2", "25/04/2024", "Thursday", "14:00 PM to 15:00 PM"))
    # exams.append(exam("Molecular and Cellular Biology (final)", "BT118","26/04/2024", "Friday", "18:00 PM to 19:00 PM"))
    exams.append(exam("Computer Aided Numerical Methods (final)", "C-306", "28/04/2024", "Sunday", "08:30 AM to 10:30 AM"))
    exams.append(exam("Computational Methods in Material Science (final)", "MS LH-3 or MS LH-2", "28/04/2024", "Sunday", "11:00 AM to 13:00 PM"))
    # exams.append(exam("Operating Systems II (final)", "Lecture Hall Complex - 03 or 04", "29/04/2024", "Monday", "14:30 PM to 16:30 PM (tentative)"))
    # exams.append(exam("Foundation in Machine Learning (final)", "LH-04 (Lecture Hall Complex)", "30/04/2024", "Tuesday", "09:30 AM to 12:00 PM"))
    # exams.append(exam("Operating Systems II (Lab)", "Auditorium", "30/04/2024", "Tuesay", "14:30 PM to 16:30 PM (tentative)"))
    
    # print(exams_schedule(exams))
    wp.sendwhatmsg_to_group_instantly("CrrCGsFykPnCm912n4S0yp", exams_schedule(exams)) 
