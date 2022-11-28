---
layout: post
title:  "Pandas review_2"
---

# Pandas review_2
    
    준비: Pandas import


```python
# 판다스, 넘파이 불러오기
import pandas as pd
import numpy as np
```

    객체 생성(Object creation): 여러 값들로 이루어진 list로부터 Series 만들기


```python
s = pd.Series([1, 3, 5, np.nan, 6, 8])

s
```




    0    1.0
    1    3.0
    2    5.0
    3    NaN
    4    6.0
    5    8.0
    dtype: float64



DataFrame 생성: _range() 함수를 이용하여 Numpy array로부터 


```python
dates = pd.date_range("20130101", periods=6)

dates
```




    DatetimeIndex(['2013-01-01', '2013-01-02', '2013-01-03', '2013-01-04',
                   '2013-01-05', '2013-01-06'],
                  dtype='datetime64[ns]', freq='D')




```python
df = pd.DataFrame(np.random.randn(6, 4), index=dates, columns=list("ABCD"))

df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>A</th>
      <th>B</th>
      <th>C</th>
      <th>D</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2013-01-01</th>
      <td>1.543271</td>
      <td>-0.854105</td>
      <td>-0.308274</td>
      <td>0.061417</td>
    </tr>
    <tr>
      <th>2013-01-02</th>
      <td>0.153895</td>
      <td>0.674997</td>
      <td>0.733301</td>
      <td>0.738170</td>
    </tr>
    <tr>
      <th>2013-01-03</th>
      <td>0.111908</td>
      <td>-0.200324</td>
      <td>0.585743</td>
      <td>1.006798</td>
    </tr>
    <tr>
      <th>2013-01-04</th>
      <td>-0.684513</td>
      <td>1.202176</td>
      <td>-0.603314</td>
      <td>-0.330621</td>
    </tr>
    <tr>
      <th>2013-01-05</th>
      <td>0.618268</td>
      <td>-0.113888</td>
      <td>0.575223</td>
      <td>0.480110</td>
    </tr>
    <tr>
      <th>2013-01-06</th>
      <td>0.442721</td>
      <td>-0.637135</td>
      <td>-0.460830</td>
      <td>1.229016</td>
    </tr>
  </tbody>
</table>
</div>



Dataframe 생성: 여러 가지 방법. 특히 Series와 유사한 구조를 지닌 Dictionary로부터 


```python
df2 = pd.DataFrame(
    {
        "A": 1.0,
        "B": pd.Timestamp("20130102"),
        "C": pd.Series(1, index=list(range(4)), dtype="float32"),
        "D": np.array([3] * 4, dtype="int32"),
        "E": pd.Categorical(["test", "train", "test", "train"]),
        "F": "foo",
    }
)


df2
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>A</th>
      <th>B</th>
      <th>C</th>
      <th>D</th>
      <th>E</th>
      <th>F</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.0</td>
      <td>2013-01-02</td>
      <td>1.0</td>
      <td>3</td>
      <td>test</td>
      <td>foo</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.0</td>
      <td>2013-01-02</td>
      <td>1.0</td>
      <td>3</td>
      <td>train</td>
      <td>foo</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.0</td>
      <td>2013-01-02</td>
      <td>1.0</td>
      <td>3</td>
      <td>test</td>
      <td>foo</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.0</td>
      <td>2013-01-02</td>
      <td>1.0</td>
      <td>3</td>
      <td>train</td>
      <td>foo</td>
    </tr>
  </tbody>
</table>
</div>




```python
A~F의 Data type은 서로 다르다.
```


      Input In [6]
        A~F의 Data type은 서로 다르다.
         ^
    SyntaxError: invalid syntax




```python
df2.dtypes
```




    A           float64
    B    datetime64[ns]
    C           float32
    D             int32
    E          category
    F            object
    dtype: object



<데이터 보기>
DataFrame.head()은 위쪽 행, DataFrame.tail()은 아래쪽 행


```python
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>A</th>
      <th>B</th>
      <th>C</th>
      <th>D</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2013-01-01</th>
      <td>1.543271</td>
      <td>-0.854105</td>
      <td>-0.308274</td>
      <td>0.061417</td>
    </tr>
    <tr>
      <th>2013-01-02</th>
      <td>0.153895</td>
      <td>0.674997</td>
      <td>0.733301</td>
      <td>0.738170</td>
    </tr>
    <tr>
      <th>2013-01-03</th>
      <td>0.111908</td>
      <td>-0.200324</td>
      <td>0.585743</td>
      <td>1.006798</td>
    </tr>
    <tr>
      <th>2013-01-04</th>
      <td>-0.684513</td>
      <td>1.202176</td>
      <td>-0.603314</td>
      <td>-0.330621</td>
    </tr>
    <tr>
      <th>2013-01-05</th>
      <td>0.618268</td>
      <td>-0.113888</td>
      <td>0.575223</td>
      <td>0.480110</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.tail(3)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>A</th>
      <th>B</th>
      <th>C</th>
      <th>D</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2013-01-04</th>
      <td>-0.684513</td>
      <td>1.202176</td>
      <td>-0.603314</td>
      <td>-0.330621</td>
    </tr>
    <tr>
      <th>2013-01-05</th>
      <td>0.618268</td>
      <td>-0.113888</td>
      <td>0.575223</td>
      <td>0.480110</td>
    </tr>
    <tr>
      <th>2013-01-06</th>
      <td>0.442721</td>
      <td>-0.637135</td>
      <td>-0.460830</td>
      <td>1.229016</td>
    </tr>
  </tbody>
</table>
</div>



출력: DataFrame.index 또는 Dataframe.columns


```python
df.index
```




    DatetimeIndex(['2013-01-01', '2013-01-02', '2013-01-03', '2013-01-04',
                   '2013-01-05', '2013-01-06'],
                  dtype='datetime64[ns]', freq='D')




```python
df.columns
```




    Index(['A', 'B', 'C', 'D'], dtype='object')



Pandas와 Numpy의 차이: 넘파이는 array 전체에 1개의 데이터 종류만 허용, Pandas는 상관없음.
                       Dataframe.to_numpy()를 사용


```python
df.to_numpy()
```




    array([[ 1.54327113, -0.85410508, -0.30827428,  0.061417  ],
           [ 0.15389464,  0.67499674,  0.73330127,  0.73816966],
           [ 0.11190845, -0.20032369,  0.58574338,  1.00679832],
           [-0.6845126 ,  1.2021763 , -0.60331372, -0.33062138],
           [ 0.61826786, -0.11388823,  0.5752233 ,  0.4801101 ],
           [ 0.44272078, -0.63713542, -0.46082959,  1.22901635]])



df2는 여러 데이터 종류로 이뤄진 DataFrame이다. 아래처럼 실행해 보면...


```python
df2.to_numpy()
```




    array([[1.0, Timestamp('2013-01-02 00:00:00'), 1.0, 3, 'test', 'foo'],
           [1.0, Timestamp('2013-01-02 00:00:00'), 1.0, 3, 'train', 'foo'],
           [1.0, Timestamp('2013-01-02 00:00:00'), 1.0, 3, 'test', 'foo'],
           [1.0, Timestamp('2013-01-02 00:00:00'), 1.0, 3, 'train', 'foo']],
          dtype=object)



Dataframe.to_numpy() 함수의 출력 결과에 index나 column label은 포함되지 않는다.

describe() 함수는 데이터를 간략하게 요약하여 보여줌


```python
df.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>A</th>
      <th>B</th>
      <th>C</th>
      <th>D</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>6.000000</td>
      <td>6.000000</td>
      <td>6.000000</td>
      <td>6.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>0.364258</td>
      <td>0.011953</td>
      <td>0.086975</td>
      <td>0.530815</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.730562</td>
      <td>0.785943</td>
      <td>0.606251</td>
      <td>0.586859</td>
    </tr>
    <tr>
      <th>min</th>
      <td>-0.684513</td>
      <td>-0.854105</td>
      <td>-0.603314</td>
      <td>-0.330621</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.122405</td>
      <td>-0.527932</td>
      <td>-0.422691</td>
      <td>0.166090</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.298308</td>
      <td>-0.157106</td>
      <td>0.133475</td>
      <td>0.609140</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>0.574381</td>
      <td>0.477775</td>
      <td>0.583113</td>
      <td>0.939641</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1.543271</td>
      <td>1.202176</td>
      <td>0.733301</td>
      <td>1.229016</td>
    </tr>
  </tbody>
</table>
</div>



데이터 변환


```python
df.T
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>2013-01-01</th>
      <th>2013-01-02</th>
      <th>2013-01-03</th>
      <th>2013-01-04</th>
      <th>2013-01-05</th>
      <th>2013-01-06</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>A</th>
      <td>1.543271</td>
      <td>0.153895</td>
      <td>0.111908</td>
      <td>-0.684513</td>
      <td>0.618268</td>
      <td>0.442721</td>
    </tr>
    <tr>
      <th>B</th>
      <td>-0.854105</td>
      <td>0.674997</td>
      <td>-0.200324</td>
      <td>1.202176</td>
      <td>-0.113888</td>
      <td>-0.637135</td>
    </tr>
    <tr>
      <th>C</th>
      <td>-0.308274</td>
      <td>0.733301</td>
      <td>0.585743</td>
      <td>-0.603314</td>
      <td>0.575223</td>
      <td>-0.460830</td>
    </tr>
    <tr>
      <th>D</th>
      <td>0.061417</td>
      <td>0.738170</td>
      <td>1.006798</td>
      <td>-0.330621</td>
      <td>0.480110</td>
      <td>1.229016</td>
    </tr>
  </tbody>
</table>
</div>



DataFrame.sort_index() sorts by an axis: 축에 의한 정렬


```python
df.sort_index(axis=1, ascending=False)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>D</th>
      <th>C</th>
      <th>B</th>
      <th>A</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2013-01-01</th>
      <td>0.061417</td>
      <td>-0.308274</td>
      <td>-0.854105</td>
      <td>1.543271</td>
    </tr>
    <tr>
      <th>2013-01-02</th>
      <td>0.738170</td>
      <td>0.733301</td>
      <td>0.674997</td>
      <td>0.153895</td>
    </tr>
    <tr>
      <th>2013-01-03</th>
      <td>1.006798</td>
      <td>0.585743</td>
      <td>-0.200324</td>
      <td>0.111908</td>
    </tr>
    <tr>
      <th>2013-01-04</th>
      <td>-0.330621</td>
      <td>-0.603314</td>
      <td>1.202176</td>
      <td>-0.684513</td>
    </tr>
    <tr>
      <th>2013-01-05</th>
      <td>0.480110</td>
      <td>0.575223</td>
      <td>-0.113888</td>
      <td>0.618268</td>
    </tr>
    <tr>
      <th>2013-01-06</th>
      <td>1.229016</td>
      <td>-0.460830</td>
      <td>-0.637135</td>
      <td>0.442721</td>
    </tr>
  </tbody>
</table>
</div>




```python
<Selection> 선택
DataFrame.at(), DataFrame.iat(), DataFrame.loc() and DataFrame.iloc()

1개의 열을 선택하면 Series가 생성된다. df["A"]와 df.A는 같은 기능
```


      Input In [17]
        <Selection> 선택
        ^
    SyntaxError: invalid syntax




```python
df["A"]
```




    2013-01-01    1.543271
    2013-01-02    0.153895
    2013-01-03    0.111908
    2013-01-04   -0.684513
    2013-01-05    0.618268
    2013-01-06    0.442721
    Freq: D, Name: A, dtype: float64



[] (__getitem__) 형식으로 선택하면 행을 분할(slicing)한다.


```python
df[0:3]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>A</th>
      <th>B</th>
      <th>C</th>
      <th>D</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2013-01-01</th>
      <td>1.543271</td>
      <td>-0.854105</td>
      <td>-0.308274</td>
      <td>0.061417</td>
    </tr>
    <tr>
      <th>2013-01-02</th>
      <td>0.153895</td>
      <td>0.674997</td>
      <td>0.733301</td>
      <td>0.738170</td>
    </tr>
    <tr>
      <th>2013-01-03</th>
      <td>0.111908</td>
      <td>-0.200324</td>
      <td>0.585743</td>
      <td>1.006798</td>
    </tr>
  </tbody>
</table>
</div>




```python
df["20130102":"20130104"]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>A</th>
      <th>B</th>
      <th>C</th>
      <th>D</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2013-01-02</th>
      <td>0.153895</td>
      <td>0.674997</td>
      <td>0.733301</td>
      <td>0.738170</td>
    </tr>
    <tr>
      <th>2013-01-03</th>
      <td>0.111908</td>
      <td>-0.200324</td>
      <td>0.585743</td>
      <td>1.006798</td>
    </tr>
    <tr>
      <th>2013-01-04</th>
      <td>-0.684513</td>
      <td>1.202176</td>
      <td>-0.603314</td>
      <td>-0.330621</td>
    </tr>
  </tbody>
</table>
</div>




```python
<Label로 선택하기>
df.loc[]와 df.iloc[]는 오직 행만, 열만, 혹은 둘 다 선택할 수 있음
df.at[]과 df.iat[]은 행과 열에서 단일값(single value)을 선택할 수 있음
```


      Input In [21]
        <Label로 선택하기>
        ^
    SyntaxError: invalid syntax




```python
df.loc[dates[0]]
```




    A    1.543271
    B   -0.854105
    C   -0.308274
    D    0.061417
    Name: 2013-01-01 00:00:00, dtype: float64



Lable로 다축(multi-axis)도 선택 가능


```python
df.loc[:, ["A", "B"]]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>A</th>
      <th>B</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2013-01-01</th>
      <td>1.543271</td>
      <td>-0.854105</td>
    </tr>
    <tr>
      <th>2013-01-02</th>
      <td>0.153895</td>
      <td>0.674997</td>
    </tr>
    <tr>
      <th>2013-01-03</th>
      <td>0.111908</td>
      <td>-0.200324</td>
    </tr>
    <tr>
      <th>2013-01-04</th>
      <td>-0.684513</td>
      <td>1.202176</td>
    </tr>
    <tr>
      <th>2013-01-05</th>
      <td>0.618268</td>
      <td>-0.113888</td>
    </tr>
    <tr>
      <th>2013-01-06</th>
      <td>0.442721</td>
      <td>-0.637135</td>
    </tr>
  </tbody>
</table>
</div>



Label 슬라이싱해서 보기(이름으로 잘랐기 때문에 양쪽 끝단에 해당하는 행이나 열까지 보임


```python
df.loc["20130102":"20130104", ["A", "B"]]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>A</th>
      <th>B</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2013-01-02</th>
      <td>0.153895</td>
      <td>0.674997</td>
    </tr>
    <tr>
      <th>2013-01-03</th>
      <td>0.111908</td>
      <td>-0.200324</td>
    </tr>
    <tr>
      <th>2013-01-04</th>
      <td>-0.684513</td>
      <td>1.202176</td>
    </tr>
  </tbody>
</table>
</div>



반환된 객체의 차원(dimension)이 줄어들 수도 있음(당연히)


```python
df.loc["20130102", ["A", "B"]]
```




    A    0.153895
    B    0.674997
    Name: 2013-01-02 00:00:00, dtype: float64



Scalar 값을 얻기 위해서 아래와 같이 실행


```python
df.loc[dates[0], "A"]
```




    1.5432711269720498




```python
# 아래와 같이 해도 동일한 결과를 얻는다.

df.at[dates[0], "A"]
```




    1.5432711269720498



스칼라 = 단일 값, 벡터화 = 통쨰로

예를 들어 merong이라는 칼럼의 값을 2배로 만들고 싶다면?
for 반복문을 사용하면 아래와 같은 코드가 나온다.

for row in range(len(df)):
    df.loc[row, 'merong'] = df.loc[row, 'merong'] * 2
    row = row + 1
    
이 방법은 비효율의 극치. 데이터프레임의 처음부터 끝까지 훑어야 한다 --> 느리다
바람직한 방법은 아래와 같음

df['merong'] = df['merong'] * 2

merong 칼럼(pandas에는 시리즈) 전체에 바로 계산을 때리는 것.
이렇게 한 방에 계산을 처리하는 방식을 벡터 연산, 혹은 벡터화 연산이라고 한다.


<위치로 선택하기> 
 DataFrame.iloc() 또는 DataFrame.at().


```python
# 3번 행

df.iloc[3]
```




    A    1.528037
    B    0.554116
    C    0.304701
    D   -0.399191
    Name: 2013-01-04 00:00:00, dtype: float64




```python
# 슬라이싱
df.iloc[3:5, 0:2]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>A</th>
      <th>B</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2013-01-04</th>
      <td>1.528037</td>
      <td>0.554116</td>
    </tr>
    <tr>
      <th>2013-01-05</th>
      <td>1.493537</td>
      <td>-0.378416</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 위치를 가리키는 정수 리스트로 선택
df.iloc[[1, 2, 4], [0, 2]]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>A</th>
      <th>C</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2013-01-02</th>
      <td>-0.712232</td>
      <td>1.418889</td>
    </tr>
    <tr>
      <th>2013-01-03</th>
      <td>1.011522</td>
      <td>0.155101</td>
    </tr>
    <tr>
      <th>2013-01-05</th>
      <td>1.493537</td>
      <td>1.370686</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 열은 놔두고 행만 슬라이싱하기
df.iloc[1:3, :]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>A</th>
      <th>B</th>
      <th>C</th>
      <th>D</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2013-01-02</th>
      <td>-0.712232</td>
      <td>-0.121785</td>
      <td>1.418889</td>
      <td>0.309798</td>
    </tr>
    <tr>
      <th>2013-01-03</th>
      <td>1.011522</td>
      <td>2.198403</td>
      <td>0.155101</td>
      <td>0.379562</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 행은 놔두고 열만 슬라이싱하기
df.iloc[:, 1:3]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>B</th>
      <th>C</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2013-01-01</th>
      <td>-0.437678</td>
      <td>2.168803</td>
    </tr>
    <tr>
      <th>2013-01-02</th>
      <td>-0.121785</td>
      <td>1.418889</td>
    </tr>
    <tr>
      <th>2013-01-03</th>
      <td>2.198403</td>
      <td>0.155101</td>
    </tr>
    <tr>
      <th>2013-01-04</th>
      <td>0.554116</td>
      <td>0.304701</td>
    </tr>
    <tr>
      <th>2013-01-05</th>
      <td>-0.378416</td>
      <td>1.370686</td>
    </tr>
    <tr>
      <th>2013-01-06</th>
      <td>2.309822</td>
      <td>0.169332</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 특정 단일값(스칼라) 얻기
df.iloc[1, 1]
```




    -0.12178522947356778




```python
# 아래와처럼 해도 동일한 결과, but 빠른 접근 가능
df.iat[1, 1]
```




    -0.12178522947356778



<Boolean 인덱싱> - 단일 칼럼의 값들로 데이터 선택하기


```python
# 칼럼 A의 값이 0보다 큰 애들만
df[df["A"] > 0]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>A</th>
      <th>B</th>
      <th>C</th>
      <th>D</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2013-01-01</th>
      <td>0.935334</td>
      <td>-0.437678</td>
      <td>2.168803</td>
      <td>-1.226609</td>
    </tr>
    <tr>
      <th>2013-01-03</th>
      <td>1.011522</td>
      <td>2.198403</td>
      <td>0.155101</td>
      <td>0.379562</td>
    </tr>
    <tr>
      <th>2013-01-04</th>
      <td>1.528037</td>
      <td>0.554116</td>
      <td>0.304701</td>
      <td>-0.399191</td>
    </tr>
    <tr>
      <th>2013-01-05</th>
      <td>1.493537</td>
      <td>-0.378416</td>
      <td>1.370686</td>
      <td>-0.405277</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 데이터프레임 전체에서 조건에 맞는 값들만 선택하기
df[df > 0]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>A</th>
      <th>B</th>
      <th>C</th>
      <th>D</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2013-01-01</th>
      <td>0.935334</td>
      <td>NaN</td>
      <td>2.168803</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2013-01-02</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.418889</td>
      <td>0.309798</td>
    </tr>
    <tr>
      <th>2013-01-03</th>
      <td>1.011522</td>
      <td>2.198403</td>
      <td>0.155101</td>
      <td>0.379562</td>
    </tr>
    <tr>
      <th>2013-01-04</th>
      <td>1.528037</td>
      <td>0.554116</td>
      <td>0.304701</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2013-01-05</th>
      <td>1.493537</td>
      <td>NaN</td>
      <td>1.370686</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2013-01-06</th>
      <td>NaN</td>
      <td>2.309822</td>
      <td>0.169332</td>
      <td>0.097655</td>
    </tr>
  </tbody>
</table>
</div>



<과제> 여기까지 내용으로 활용하여 연습문제를 3개씩 만들어 올 것. 
       문제와 풀이과정 및 해답 전부를 깃허브 또는 깃허브 블로그에 포스팅


isin() 함수로 필터링


```python
df2 = df.copy()
df2["E"] = ["one", "one", "two", "three", "four", "three"]
df2
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>A</th>
      <th>B</th>
      <th>C</th>
      <th>D</th>
      <th>E</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2013-01-01</th>
      <td>0.935334</td>
      <td>-0.437678</td>
      <td>2.168803</td>
      <td>-1.226609</td>
      <td>one</td>
    </tr>
    <tr>
      <th>2013-01-02</th>
      <td>-0.712232</td>
      <td>-0.121785</td>
      <td>1.418889</td>
      <td>0.309798</td>
      <td>one</td>
    </tr>
    <tr>
      <th>2013-01-03</th>
      <td>1.011522</td>
      <td>2.198403</td>
      <td>0.155101</td>
      <td>0.379562</td>
      <td>two</td>
    </tr>
    <tr>
      <th>2013-01-04</th>
      <td>1.528037</td>
      <td>0.554116</td>
      <td>0.304701</td>
      <td>-0.399191</td>
      <td>three</td>
    </tr>
    <tr>
      <th>2013-01-05</th>
      <td>1.493537</td>
      <td>-0.378416</td>
      <td>1.370686</td>
      <td>-0.405277</td>
      <td>four</td>
    </tr>
    <tr>
      <th>2013-01-06</th>
      <td>-0.326543</td>
      <td>2.309822</td>
      <td>0.169332</td>
      <td>0.097655</td>
      <td>three</td>
    </tr>
  </tbody>
</table>
</div>




```python
df2[df2["E"].isin(["two", "four"])]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>A</th>
      <th>B</th>
      <th>C</th>
      <th>D</th>
      <th>E</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2013-01-03</th>
      <td>1.011522</td>
      <td>2.198403</td>
      <td>0.155101</td>
      <td>0.379562</td>
      <td>two</td>
    </tr>
    <tr>
      <th>2013-01-05</th>
      <td>1.493537</td>
      <td>-0.378416</td>
      <td>1.370686</td>
      <td>-0.405277</td>
      <td>four</td>
    </tr>
  </tbody>
</table>
</div>



## Setting 

#### 어떤 값을 넣는 것을 


```python
s1 = pd.Series([1, 2, 3, 4, 5, 6], index=pd.date_range("20130102", periods=6))
s1
```




    2013-01-02    1
    2013-01-03    2
    2013-01-04    3
    2013-01-05    4
    2013-01-06    5
    2013-01-07    6
    Freq: D, dtype: int64




```python
df["F"] = s1
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>A</th>
      <th>B</th>
      <th>C</th>
      <th>D</th>
      <th>F</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2013-01-01</th>
      <td>0.935334</td>
      <td>-0.437678</td>
      <td>2.168803</td>
      <td>-1.226609</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2013-01-02</th>
      <td>-0.712232</td>
      <td>-0.121785</td>
      <td>1.418889</td>
      <td>0.309798</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2013-01-03</th>
      <td>1.011522</td>
      <td>2.198403</td>
      <td>0.155101</td>
      <td>0.379562</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>2013-01-04</th>
      <td>1.528037</td>
      <td>0.554116</td>
      <td>0.304701</td>
      <td>-0.399191</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>2013-01-05</th>
      <td>1.493537</td>
      <td>-0.378416</td>
      <td>1.370686</td>
      <td>-0.405277</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>2013-01-06</th>
      <td>-0.326543</td>
      <td>2.309822</td>
      <td>0.169332</td>
      <td>0.097655</td>
      <td>5.0</td>
    </tr>
  </tbody>
</table>
</div>



레이블로 value setting


```python
df.at[dates[0], "A"] = 0
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>A</th>
      <th>B</th>
      <th>C</th>
      <th>D</th>
      <th>F</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2013-01-01</th>
      <td>0.000000</td>
      <td>-0.437678</td>
      <td>2.168803</td>
      <td>-1.226609</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2013-01-02</th>
      <td>-0.712232</td>
      <td>-0.121785</td>
      <td>1.418889</td>
      <td>0.309798</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2013-01-03</th>
      <td>1.011522</td>
      <td>2.198403</td>
      <td>0.155101</td>
      <td>0.379562</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>2013-01-04</th>
      <td>1.528037</td>
      <td>0.554116</td>
      <td>0.304701</td>
      <td>-0.399191</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>2013-01-05</th>
      <td>1.493537</td>
      <td>-0.378416</td>
      <td>1.370686</td>
      <td>-0.405277</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>2013-01-06</th>
      <td>-0.326543</td>
      <td>2.309822</td>
      <td>0.169332</td>
      <td>0.097655</td>
      <td>5.0</td>
    </tr>
  </tbody>
</table>
</div>



위치로 값 넣기


```python
df.iat[0, 1] = 0
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>A</th>
      <th>B</th>
      <th>C</th>
      <th>D</th>
      <th>F</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2013-01-01</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>2.168803</td>
      <td>-1.226609</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2013-01-02</th>
      <td>-0.712232</td>
      <td>-0.121785</td>
      <td>1.418889</td>
      <td>0.309798</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2013-01-03</th>
      <td>1.011522</td>
      <td>2.198403</td>
      <td>0.155101</td>
      <td>0.379562</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>2013-01-04</th>
      <td>1.528037</td>
      <td>0.554116</td>
      <td>0.304701</td>
      <td>-0.399191</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>2013-01-05</th>
      <td>1.493537</td>
      <td>-0.378416</td>
      <td>1.370686</td>
      <td>-0.405277</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>2013-01-06</th>
      <td>-0.326543</td>
      <td>2.309822</td>
      <td>0.169332</td>
      <td>0.097655</td>
      <td>5.0</td>
    </tr>
  </tbody>
</table>
</div>



넘파이 어레이에 assign(할당)해서 setting


```python
df.loc[:, "D"] = np.array([5] * len(df))
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>A</th>
      <th>B</th>
      <th>C</th>
      <th>D</th>
      <th>F</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2013-01-01</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>2.168803</td>
      <td>5</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2013-01-02</th>
      <td>-0.712232</td>
      <td>-0.121785</td>
      <td>1.418889</td>
      <td>5</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2013-01-03</th>
      <td>1.011522</td>
      <td>2.198403</td>
      <td>0.155101</td>
      <td>5</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>2013-01-04</th>
      <td>1.528037</td>
      <td>0.554116</td>
      <td>0.304701</td>
      <td>5</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>2013-01-05</th>
      <td>1.493537</td>
      <td>-0.378416</td>
      <td>1.370686</td>
      <td>5</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>2013-01-06</th>
      <td>-0.326543</td>
      <td>2.309822</td>
      <td>0.169332</td>
      <td>5</td>
      <td>5.0</td>
    </tr>
  </tbody>
</table>
</div>






```python
df2 = df.copy()
df2[df2 > 0] = -df2
df2
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>A</th>
      <th>B</th>
      <th>C</th>
      <th>D</th>
      <th>F</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2013-01-01</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>-2.168803</td>
      <td>-5</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2013-01-02</th>
      <td>-0.712232</td>
      <td>-0.121785</td>
      <td>-1.418889</td>
      <td>-5</td>
      <td>-1.0</td>
    </tr>
    <tr>
      <th>2013-01-03</th>
      <td>-1.011522</td>
      <td>-2.198403</td>
      <td>-0.155101</td>
      <td>-5</td>
      <td>-2.0</td>
    </tr>
    <tr>
      <th>2013-01-04</th>
      <td>-1.528037</td>
      <td>-0.554116</td>
      <td>-0.304701</td>
      <td>-5</td>
      <td>-3.0</td>
    </tr>
    <tr>
      <th>2013-01-05</th>
      <td>-1.493537</td>
      <td>-0.378416</td>
      <td>-1.370686</td>
      <td>-5</td>
      <td>-4.0</td>
    </tr>
    <tr>
      <th>2013-01-06</th>
      <td>-0.326543</td>
      <td>-2.309822</td>
      <td>-0.169332</td>
      <td>-5</td>
      <td>-5.0</td>
    </tr>
  </tbody>
</table>
</div>



# Missing Data
판다스에서는 np.nan을 쓴다.
특정 축에 index를 변경/추가/삭제하는 것을 reindexing이라고 한다.


```python
df = pd.DataFrame(np.random.randn(5, 3), columns=['C1', 'C2', 'C3'])
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>C1</th>
      <th>C2</th>
      <th>C3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.185509</td>
      <td>0.404934</td>
      <td>0.292162</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-0.232402</td>
      <td>0.356683</td>
      <td>0.620492</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-0.435678</td>
      <td>-0.462780</td>
      <td>-1.226629</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-0.748518</td>
      <td>-1.008454</td>
      <td>0.767842</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-2.378031</td>
      <td>0.308627</td>
      <td>-1.084080</td>
    </tr>
  </tbody>
</table>
</div>




```python

```


```python

```


```python

```
