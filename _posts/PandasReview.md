Pandas와 Numpy의 차이는 표 하나에 넣을 수 있는 데이터의 종류가 Numpy는 한 종류밖에 안된다는것이다.     
    
    스칼라 in pandas is 단일 값(single value)
    Pandas에서 칼럼을 Series라고 함
    Series는 스칼라의 컨테이너, DataFrame은 Series의 컨테이너


```python
import pandas as pd
import numpy as np
```


```python
dates = pd.date_range("20130101", periods = 6)

df = pd.DataFrame(np.random.randn(6, 4), index = dates, columns = list("ABCD"))

df.describe() #DF 각종 정보를 보여주는 명령어
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
      <td>-0.071954</td>
      <td>-0.330968</td>
      <td>0.045913</td>
      <td>-0.215833</td>
    </tr>
    <tr>
      <th>std</th>
      <td>1.393847</td>
      <td>0.667593</td>
      <td>0.899865</td>
      <td>0.504936</td>
    </tr>
    <tr>
      <th>min</th>
      <td>-1.129178</td>
      <td>-1.108157</td>
      <td>-0.780620</td>
      <td>-1.003256</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>-0.957649</td>
      <td>-0.820386</td>
      <td>-0.609902</td>
      <td>-0.480545</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>-0.640074</td>
      <td>-0.357184</td>
      <td>-0.154037</td>
      <td>-0.150465</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>0.226650</td>
      <td>-0.002619</td>
      <td>0.346326</td>
      <td>0.139968</td>
    </tr>
    <tr>
      <th>max</th>
      <td>2.525551</td>
      <td>0.695932</td>
      <td>1.612914</td>
      <td>0.362984</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.T #index와 column을 바꾸는 명령어
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
      <td>-0.870391</td>
      <td>-1.129178</td>
      <td>0.438786</td>
      <td>-0.986736</td>
      <td>2.525551</td>
      <td>-0.409757</td>
    </tr>
    <tr>
      <th>B</th>
      <td>-1.108157</td>
      <td>0.695932</td>
      <td>0.087535</td>
      <td>-0.946752</td>
      <td>-0.441286</td>
      <td>-0.273081</td>
    </tr>
    <tr>
      <th>C</th>
      <td>0.172514</td>
      <td>-0.653007</td>
      <td>0.404263</td>
      <td>1.612914</td>
      <td>-0.480588</td>
      <td>-0.780620</td>
    </tr>
    <tr>
      <th>D</th>
      <td>-1.003256</td>
      <td>0.362984</td>
      <td>-0.266835</td>
      <td>-0.551782</td>
      <td>-0.034096</td>
      <td>0.197989</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.to_numpy() #pandas data를 numpy data로 바꿔주는 명령어
```




    array([[-0.87039053, -1.10815664,  0.17251412, -1.00325603],
           [-1.12917793,  0.6959319 , -0.65300695,  0.3629838 ],
           [ 0.43878621,  0.0875347 ,  0.40426278, -0.26683455],
           [-0.98673552, -0.9467524 ,  1.61291355, -0.55178221],
           [ 2.52555087, -0.44128583, -0.48058801, -0.03409609],
           [-0.40975728, -0.27308123, -0.78062043,  0.19798949]])




```python
df.sort_index(axis=1, ascending=False) #index에(axis=1은 행, 0은 열) 따른 정렬을 오름차순을 false로 함
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
      <td>-1.003256</td>
      <td>0.172514</td>
      <td>-1.108157</td>
      <td>-0.870391</td>
    </tr>
    <tr>
      <th>2013-01-02</th>
      <td>0.362984</td>
      <td>-0.653007</td>
      <td>0.695932</td>
      <td>-1.129178</td>
    </tr>
    <tr>
      <th>2013-01-03</th>
      <td>-0.266835</td>
      <td>0.404263</td>
      <td>0.087535</td>
      <td>0.438786</td>
    </tr>
    <tr>
      <th>2013-01-04</th>
      <td>-0.551782</td>
      <td>1.612914</td>
      <td>-0.946752</td>
      <td>-0.986736</td>
    </tr>
    <tr>
      <th>2013-01-05</th>
      <td>-0.034096</td>
      <td>-0.480588</td>
      <td>-0.441286</td>
      <td>2.525551</td>
    </tr>
    <tr>
      <th>2013-01-06</th>
      <td>0.197989</td>
      <td>-0.780620</td>
      <td>-0.273081</td>
      <td>-0.409757</td>
    </tr>
  </tbody>
</table>
</div>



Code: code
MarkDown: style text
Raw NBConvert: simple text
Heading: title
