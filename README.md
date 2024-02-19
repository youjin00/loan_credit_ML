데이콘의 '고객 ID로 대출등급' 예측하기 데이터 자료를 활용했습니다.
## EDA

### 1. 데이터 변수 이해

clean data로 null 값이 존재하지 않음

- 범주형 데이터 : 6개의 피쳐
    - ID : 총 96294개의 아이디가 존재함
    - 대출기간 및 근로기간 : 문자형 데이터와 합쳐져서 분리할 필요가 있음
        - 대출기간 및 근로기간을 수치형 데이터로 전환해줌
    - 주택소유상태 : 담보 대출
        - 담보대출의 경우, 경제적 가치가 있는 자산을 보증으로 두는 것을 의미함
        - 담보가 없어도 안정적인 직장이 있다면 신용 대출이 가능
            - Mortgage : 담보 대출 → 소득이 낮을 수도 있음
                - 그렇다면 담보 대출과 소득의 연관관계에 대해 알아볼 필요가 있음
            - Rent
            - Own
            - Any
    - 대출목적
    - 대출등급
- 수치형 데이터 : 9개의 피쳐 → 총 15개의 피쳐 값이 존재함
    - describe 함수로 뽑힌 데이터를 수치형 데이터로 간주함
    - 대출금액
    - 연간소득
    - 부채_대비_소득_비율
    - 총계좌수
    - 최근 2년간 연체 횟수
    - 총상환원금
    - 총상환이자 : 분할 상환 형식임을 보여줌
    - 총연체금액
    - 연체계좌 수
1. 수치형 데이터가 확실한 대출기간과 근로기간을 바꿔줌

```python
df_train = train.copy()
df_test = test.copy()
df_train['대출기간'] = df_train['대출기간'].str.replace('[A-Za-z]+', '').astype(int)
df_train['근로기간'] = df_train['근로기간'].str.replace('[A-Za-z]+', '')
df_train['근로기간'] = df_train['근로기간'].str.replace(' ', '')
df_train['근로기간'] = df_train['근로기간'].replace(['<1'], '0')
df_train['근로기간'] = df_train['근로기간'].replace(['10+'], '10')
df_train['대출기간'] = df_train['대출기간'].replace([36], 3)
df_train['대출기간'] = df_train['대출기간'].replace([60], 5)

df_train.head()
```

1. 데이터 feature 간의 상관관계를 이해하기 위해서 heatmap을 활용함

![newplot.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/60c516c3-f4f5-4664-a9b3-c5fdfce32594/d4b3ff9a-7855-4229-a1fb-96eee577330e/newplot.png)

- 대출금액이 높을수록, 대출기간, 연간소득, 총계좌수, 총상환이자, 총상환원금이 많음
    - Feature 값이 많으므로, background 지식을 활용해서 데이터를 걸러내도록 함
    - 대출등급은 신용점수와 연관성이 높은데, 이러한 신용점수에 따라 산정되는 경우가 많으므로 비슷한 요인을 중점으로 살펴보기로 함
    - 일반적으로 상환이력 정보, 부채 수준에 따라 신용 등급이 산정됨
        - 상환 이력 정보 : 빚을 제때 갚았는지, 과거 채무상환을 미룬 적이 있는지 등의 연체 정보
        - 부채 수준 : 대출규모 및 신용 카드 이용금액 등 현재의 채무 보유 수준
        - 신용 거래 기간 : 대출 거래 기간
        - 신용형태 정보 : 대출 거래 기관 수, 신용거래의 종류 및 형태
        - 자산 대비 소득 비율이 높을수록, 신용평점에 부정적인 영향을 줄 가능성이 높음
        
        → 연간소득을 제외한 모든 feature 값들이 대출등급과 양의 관계에 있을 것이라 예상할 수 있음
        
    - 대출의 경우, 담보대출과 신용대출로 나뉨
        - 신용대출은 앞서 언급한 신용 등급의 영향을 받고, 담보 대출의 경우 주택 등을 저당으로 대출을 받음 → 따라서 우선 저당이면서 소득이 높은 사람의 대출 등급과 저당이면서 소득이 낮은 사람의 대출등급을 파악하도록 함
        - 따라서, 담보대출과 신용대출을 구분해서 대출등급을 파악하기로 함
    
    ```python
    def mortgage_data(feature) :
        Mortgage = df_train[df_train['주택소유상태'] == 'MORTGAGE'][feature].value_counts()
        Credit = df_train[df_train['주택소유상태'] != 'MORTGAGE'][feature].value_counts()
        
        graph = pd.DataFrame([Mortgage, Credit])
        graph.index = ['Mortgage', 'Credit']
    
        return graph
    ```
    
    - 간단한 함수를 만들어서 파악함
    
    ![newplot (1).png](https://prod-files-secure.s3.us-west-2.amazonaws.com/60c516c3-f4f5-4664-a9b3-c5fdfce32594/a53bacd9-4db1-4564-b2a7-3192e64d6e4a/newplot_(1).png)
    
    - 예상한대로, mortgage로 대출을 받았을 경우 대출등급이 높은 것을 알 수 있음
    - 위와 같은 분류의 경우 credit은 연간소득과 근로기간 영향을 많이 받을 것으로 예상됨
        - 대출등급에 공통적으로 영향을 주는 요인은 대출금액, 연간소득, 부채_대비_소득_비율, 최근 2년간 연체 횟수, 총상환원금, 연체금액, 상환이자 등임
        - Credit 내에서 분류를 나누려면, 그 중에서 연간소득, 근로기간이 가장 큰 영향을 줄 것이라 예상됨
            
            → 연간소득과 근로기간이 많고 길수록 직업의 안정성이 보장되어 대출등급이 높을 것임
            
    
    1. 신용대출 등급 분석
        
        1) 등급이 높을수록 대출기간이 3년이 더 많은 경향이 있음
        
        2) 등급이 높을수록 연간소득이 높을 것이라 예상되기 때문에 함께 시각화 처리를 하도록함
        
        - 연간소득
            - A등급의 경우, 절반이 평균 이상의 연간 소득을 가짐
            - 즉, 연간 소득이 평균 미만인 경우 B~G 등급일 확률이 큼
        - 대출기간
            - A~C 등급인 경우, 대출기간이 3년일 확률이 큼
        - 대출금액
            - 평균 대출 금액 이상인 경우, 대출등급이 E, F, G일 확률이 큼
            - 특히 G등급의 경우, 대부분이 평균 대출 금액 이상을 대출함
        - 부채 대비 소득 비율
            - 비율이 높을 수록 부채 상환 능력이 낮아지는 의미임
            - A,B 등급의 경우 평균보다 부채 대비 소득 비율이 낮음
            - C~G 등급의 경우 평균보다 부채 대비 소득 비율이 높음
        - 총계좌수는 중요성이 떨어지기 때문에 고려하지 않기로 함
        - 연체계좌수와 최근2년간연체횟수 간의 관계를 파악해보면 0.13으로 꽤 관련이 있는 것을 알 수 있음
            - 최근 2년간 연체횟수는 중요한 의미를 못 찾았음
        
        → 신용대출에서는 전체적으로 연체 계좌나 연체 횟수가 적음
        
        - 총상환금액 : A와 B등급의 경우 총상환원금이 중위값 이상이 많음
        - 총이자상환금액 : A와 B등급의 경우 중위값 이하가 많음 → 상환금액이 많기 때문에 이자가 적게 붙었을 것이라 유추 가능
    
    ```python
    high_fair = credit[credit['연간소득'] >= 1.440000e+07]['대출등급'].value_counts() #평균 대출금액 이상
    low_fair = credit[credit['연간소득'] < 8.138032e+07]['대출등급'].value_counts()
    
    df1 = pd.concat([high_fair, low_fair], axis=1, keys=['high_fair', 'low_fair'])
    df1.iplot(kind='bar', barmode='stack')
    
    high_ratio = credit[credit['부채_대비_소득_비율'] >= 19.317551]['대출등급'].value_counts() #평균 대출 금액 이상
    low_ratio = credit[credit['부채_대비_소득_비율'] < 19.317551]['대출등급'].value_counts()
    
    df2 = pd.concat([high_ratio, low_ratio], axis=1, keys=['high_ratio', 'low_ratio'])
    df2.iplot(kind='bar', barmode='stack')
    
    high_overdue = credit[credit['총연체금액'] >= 59]['대출등급'].value_counts() #연체 0번 초과
    low_overdue = credit[credit['총연체금액'] < 59]['대출등급'].value_counts() #연체 0번 이하
    
    df3 = pd.concat([high_overdue, low_overdue], axis=1, keys=['high_overdue', 'low_overdue'])
    df3.iplot(kind='bar', barmode='stack')
    
    high_repay = credit[credit['총상환원금'] >= 5.552160e+05]['대출등급'].value_counts() #연체 0번 초과
    low_repay = credit[credit['총상환원금'] < 5.552160e+05]['대출등급'].value_counts() 
    
    df4 = pd.concat([high_repay, low_repay], axis=1, keys=['high_repay', 'low_repay'])
    df4.iplot(kind='bar', barmode='stack')
    
    high_interest_repay = credit[credit['총상환이자'] >= 2.625960e+05]['대출등급'].value_counts() #연체 0번 초과
    low_interest_repay = credit[credit['총상환이자'] < 2.625960e+05]['대출등급'].value_counts() 
    
    df5 = pd.concat([high_interest_repay, low_interest_repay], axis=1, keys=['high_interest_repay', 'low_interest_repay'])
    df5.iplot(kind='bar', barmode='stack')
    ```
    
    1. 담보대출 등급 분석
    - 대출금액이 신용대출보다 더 많음 → 담보대출이 신용 등급이 높기 때문이라 예측해볼 수 있음
    - 연간소득
        - A등급인 경우, 연간 소득이 평균 이상임
        - 연간소득이 평균 이상 + 담보대출 = A등급 확률이 큼
    - 부채 비율
        - A,B 등급인 경우 비율이 낮음
    - 연체 요소는 여기서도 파악이 어려움 → 연체 요소는 그렇게 뚜렷하게 구별되는 요소가 아님을 알 수 있음
    - 상환 금액 : A등급인 경우 평균보다 상환 금액이 높음
    - 이자 금액 : A,B등급은 이자 상환 비율이 낮음
    
    ```python
    high_fair = mortgage[mortgage['연간소득'] >= 1.065846e+08]['대출등급'].value_counts() #평균 대출금액 이상
    low_fair = mortgage[mortgage['연간소득'] < 1.065846e+08]['대출등급'].value_counts()
    
    df1 = pd.concat([high_fair, low_fair], axis=1, keys=['high_fair', 'low_fair'])
    df1.iplot(kind='bar', barmode='stack')
    
    high_ratio = mortgage[mortgage['부채_대비_소득_비율'] >= 19.442180]['대출등급'].value_counts() #평균 대출 금액 이상
    low_ratio = mortgage[mortgage['부채_대비_소득_비율'] < 19.442180]['대출등급'].value_counts()
    
    df2 = pd.concat([high_ratio, low_ratio], axis=1, keys=['high_ratio', 'low_ratio'])
    df2.iplot(kind='bar', barmode='stack')
    
    high_overdue = mortgage[mortgage['총연체금액'] >= 49]['대출등급'].value_counts() #연체 0번 초과
    low_overdue = mortgage[mortgage['총연체금액'] < 49]['대출등급'].value_counts() #연체 0번 이하
    
    df3 = pd.concat([high_overdue, low_overdue], axis=1, keys=['high_overdue', 'low_overdue'])
    df3.iplot(kind='bar', barmode='stack')
    
    high_repay = mortgage[mortgage['총상환원금'] >= 6.601680e+05]['대출등급'].value_counts() #연체 0번 초과
    low_repay = mortgage[mortgage['총상환원금'] < 6.601680e+05]['대출등급'].value_counts() 
    
    df4 = pd.concat([high_repay, low_repay], axis=1, keys=['high_repay', 'low_repay'])
    df4.iplot(kind='bar', barmode='stack')
    
    high_interest_repay = mortgage[mortgage['총상환이자'] >= 3.147720e+05]['대출등급'].value_counts() #연체 0번 초과
    low_interest_repay = mortgage[mortgage['총상환이자'] < 3.147720e+05]['대출등급'].value_counts() 
    
    df5 = pd.concat([high_interest_repay, low_interest_repay], axis=1, keys=['high_interest_repay', 'low_interest_repay'])
    df5.iplot(kind='bar', barmode='stack')
    ```
    
    ## Feature Eng
    
    - 대출등급이 높, 중간, 낮음 세 분류로 진행
    - 대출기간 (0 : 3년, 1: 5년)
        - 대출등급 : 0
        - 대출등급 low : 1
    
    - 부채_대비_소득_비율
        - 부채대비소득비율이 3이면 대출등급이 낮을 확률이 큼
    
    - 대출목적이 1이면 대출등급이 높을 확률이 큼
    - 총상환이자가 1이면 대출등급이 높을 확률이 큼
