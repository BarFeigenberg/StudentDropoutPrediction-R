### Dataset overview
All data collected is published on: https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success

The dataset contains information on over 4,000 students from a specific higher education institution in Portugal,aiming to analyze and predict student dropout.
It includes diverse demographic, academic, and financial features such as age, gender, course of study, admission grades, debt status etc.
The target variable represents each student’s academic outcome — either Graduate or Dropout (with current students, labeled as Enrolled). 
This dataset offers a comprehensive basis for identifying key factors influencing student success and retention.

### Colomns description:

1. Marital Status – (categorial int) The student's marital status (e.g., single, married).   

2. Application mode – (categorial int) The method used to apply for admission (e.g., national exam, transfer).   

3. Application order – (categorial int) The priority/rank of the course in the student's application choices.   

4. Course – (categorial int) The specific academic program the student enrolled in.   

5. Daytime/evening attendance – (binary int) Whether the student attends classes during the day or in the evening.   

6. Previous qualification – (categorial int) The type of qualification the student had before entering this course (e.g., high school, bachelor’s).   

7. Previous qualification (grade) – (int) The grade obtained in the previous qualification.   

8. Nacionality – (categorial int) The student’s nationality.   

9. Mother's qualification – (categorial int) The highest educational level achieved by the student’s mother.   

10. Father's qualification – (categorial int) The highest educational level achieved by the student’s father.   

11. Mother's occupation – (categorial int) The job/profession of the student’s mother.   

12. Father's occupation – (categorial int) The job/profession of the student’s father.   

13. Admission grade – (float) The grade/score used for admission into the current program.   

14. Displaced – (binary int) Whether the student was displaced from their original location. 

15. Educational special needs – (binary int) Whether the student has special educational needs.   

16. Debtor – (binary int) Whether the student has unpaid debts to the institution.   

17. Tuition fees up to date – (binary int) Whether  student is current with tuition payments.   

18. Gender – (binary int) The student’s gender male=1, female=0.   

19. Scholarship holder – (binary int) Whether the student holds a scholarship.   

20. Age at enrolment – (int) The age of the student at the time of enrolment.   

21. International – (binary int) Whether the student is considered an international student.   

22. Curricular units 1st sem (credited) – (int) Number of courses in the 1st semester that were credited (i.e., transferred from previous study).   

23. Curricular units 1st sem (enrolled) – (int) Number of courses the student enrolled in during the 1st semester.   

24. Curricular units 1st sem (evaluations) – (int) Number of courses in the 1st semester in which the student was evaluated (took exams or was graded).   

25. Curricular units 1st sem (approved) – (int) Number of courses the student passed in the 1st semester.   

26. Curricular units 1st sem (grade) – (int) Average grade of courses in the 1st semester.   

27. Curricular units 1st sem (without evaluations) – (int) Number of courses the student enrolled in but was not evaluated in (missed exams, etc.).   

28. Curricular units 2nd sem (credited) – (int) Same as above, but for the 2nd semester.   

29. Curricular units 2nd sem (enrolled) – (int) Courses enrolled in during the 2nd semester.   

30. Curricular units 2nd sem (evaluations) – (int) Courses evaluated in the 2nd semester.   

31. Curricular units 2nd sem (approved) – (int) Courses passed in the 2nd semester.   

32. Curricular units 2nd sem (grade) – (int) Average grade of the 2nd semester.   

33. Curricular units 2nd sem (without evaluations) – (int) Courses in the 2nd semester without evaluations.   

34. Unemployment rate – (float) The national unemployment rate at the time of enrolment.   

35. Inflation rate – (float) The national inflation rate at the time of enrolment.   

36. GDP – (float) National gross domestic product at the time of enrolment.   

37. target – (categorial) The final outcome for the student: whether they Dropped out, are still Enrolled, or Graduated.   