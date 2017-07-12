

```python
import pandas as pd
import numpy as np
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
frames = [train, test]
data = pd.concat(frames)
data.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>1stFlrSF</th>
      <th>2ndFlrSF</th>
      <th>3SsnPorch</th>
      <th>Alley</th>
      <th>BedroomAbvGr</th>
      <th>BldgType</th>
      <th>BsmtCond</th>
      <th>BsmtExposure</th>
      <th>BsmtFinSF1</th>
      <th>BsmtFinSF2</th>
      <th>...</th>
      <th>SaleType</th>
      <th>ScreenPorch</th>
      <th>Street</th>
      <th>TotRmsAbvGrd</th>
      <th>TotalBsmtSF</th>
      <th>Utilities</th>
      <th>WoodDeckSF</th>
      <th>YearBuilt</th>
      <th>YearRemodAdd</th>
      <th>YrSold</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>856</td>
      <td>854</td>
      <td>0</td>
      <td>NaN</td>
      <td>3</td>
      <td>1Fam</td>
      <td>TA</td>
      <td>No</td>
      <td>706.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>WD</td>
      <td>0</td>
      <td>Pave</td>
      <td>8</td>
      <td>856.0</td>
      <td>AllPub</td>
      <td>0</td>
      <td>2003</td>
      <td>2003</td>
      <td>2008</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1262</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>3</td>
      <td>1Fam</td>
      <td>TA</td>
      <td>Gd</td>
      <td>978.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>WD</td>
      <td>0</td>
      <td>Pave</td>
      <td>6</td>
      <td>1262.0</td>
      <td>AllPub</td>
      <td>298</td>
      <td>1976</td>
      <td>1976</td>
      <td>2007</td>
    </tr>
    <tr>
      <th>2</th>
      <td>920</td>
      <td>866</td>
      <td>0</td>
      <td>NaN</td>
      <td>3</td>
      <td>1Fam</td>
      <td>TA</td>
      <td>Mn</td>
      <td>486.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>WD</td>
      <td>0</td>
      <td>Pave</td>
      <td>6</td>
      <td>920.0</td>
      <td>AllPub</td>
      <td>0</td>
      <td>2001</td>
      <td>2002</td>
      <td>2008</td>
    </tr>
    <tr>
      <th>3</th>
      <td>961</td>
      <td>756</td>
      <td>0</td>
      <td>NaN</td>
      <td>3</td>
      <td>1Fam</td>
      <td>Gd</td>
      <td>No</td>
      <td>216.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>WD</td>
      <td>0</td>
      <td>Pave</td>
      <td>7</td>
      <td>756.0</td>
      <td>AllPub</td>
      <td>0</td>
      <td>1915</td>
      <td>1970</td>
      <td>2006</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1145</td>
      <td>1053</td>
      <td>0</td>
      <td>NaN</td>
      <td>4</td>
      <td>1Fam</td>
      <td>TA</td>
      <td>Av</td>
      <td>655.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>WD</td>
      <td>0</td>
      <td>Pave</td>
      <td>9</td>
      <td>1145.0</td>
      <td>AllPub</td>
      <td>192</td>
      <td>2000</td>
      <td>2000</td>
      <td>2008</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 81 columns</p>
</div>




```python
#data['YearBuilt'] = data['YearBuilt'].apply(lambda x: 2017 - x)

```


```python
#pd.value_counts(data['Alley'].values, sort=True)
#data = data.drop('Alley')
```


```python
pd.value_counts(data['BldgType'].values, sort=True)

```




    1Fam      2425
    TwnhsE     227
    Duplex     109
    Twnhs       96
    2fmCon      62
    dtype: int64




```python
pd.value_counts(data['BsmtFinType1'].values, sort=True)

```




    Unf    851
    GLQ    849
    ALQ    429
    Rec    288
    BLQ    269
    LwQ    154
    dtype: int64




```python
pd.value_counts(data['BsmtCond'].values, sort=True)
```




    TA    2606
    Gd     122
    Fa     104
    Po       5
    dtype: int64




```python
#data= data.drop('BsmtCond')
data = data.drop('BsmtExposure')
data = data.drop('BsmtQual')
```


```python
pd.value_counts(data['CentralAir'].values, sort=True)
data['CentralAir'] = pd.get_dummies(data['CentralAir'], drop_first=True)
```


```python
pd.value_counts(data['Condition1'].values, sort=True)
```




    Norm      2511
    Feedr      164
    Artery      92
    RRAn        50
    PosN        39
    RRAe        28
    PosA        20
    RRNn         9
    RRNe         6
    dtype: int64




```python
pd.value_counts(data['Electrical'].values, sort=True)
```




    SBrkr    2671
    FuseA     188
    FuseF      50
    FuseP       8
    Mix         1
    dtype: int64




```python
pd.value_counts(data['ExterCond'].values, sort=True)
data = data.drop('ExterQual')
```


```python
pd.value_counts(data['Exterior1st'])
```




    VinylSd    1025
    MetalSd     450
    HdBoard     442
    Wd Sdng     411
    Plywood     221
    CemntBd     126
    BrkFace      87
    WdShing      56
    AsbShng      44
    Stucco       43
    BrkComm       6
    AsphShn       2
    Stone         2
    CBlock        2
    ImStucc       1
    Name: Exterior1st, dtype: int64




```python
pd.value_counts(data['Fence'].values, sort=True)
#data = data.drop('Fence')
```




    MnPrv    329
    GdPrv    118
    GdWo     112
    MnWw      12
    dtype: int64




```python
pd.value_counts(data['Functional'].values, sort=True)
```




    Typ     2717
    Min2      70
    Min1      65
    Mod       35
    Maj1      19
    Maj2       9
    Sev        2
    dtype: int64




```python
pd.value_counts(data['GarageCond'].values, sort=True)
data['GarageCond'] = data['GarageCond'].fillna('No')
data['GarageCond'] = data['GarageCond'].apply(lambda x: x.replace(x, 'Yes') if x != 'No' else x)
pd.value_counts(data['GarageCond'].values, sort=True)
```




    Yes    2760
    No      159
    dtype: int64




```python
data= data.drop('GarageFinish')
#data= data.drop('GarageQual')
```


```python
pd.value_counts(data['Heating'].values, sort=True)
#data['HeatingQC'] = pd.get_dummies(data['HeatingQC'], )
```




    GasA     2874
    GasW       27
    Grav        9
    Wall        6
    OthW        2
    Floor       1
    dtype: int64




```python
pd.value_counts(data['KitchenQual'].values, sort=False)

```




    Gd    1151
    Ex     205
    Fa      70
    TA    1492
    dtype: int64




```python
pd.value_counts(data['LandContour'].values, sort=True)
```




    Lvl    2622
    HLS     120
    Bnk     117
    Low      60
    dtype: int64




```python
#pd.value_counts(data['PavedDrive'].values, sort=True)
#data['PavedDrive'] = data['PavedDrive'].apply(lambda x: x.replace(x, 'no') if x != 'Y' else x)
#pd.value_counts(data['PavedDrive'].values, sort=True)
```


```python
pd.value_counts(data['PoolQC'].values, sort=True)
data['PoolQC']= data['PoolQC'].fillna('No')
data['PoolQC'] = data['PoolQC'].apply(lambda x: x.replace(x, 'Yes') if x == 'Ex' or  x == 'Gd' or x == 'Fa' else x)
pd.value_counts(data['PoolQC'].values, sort=True)
```




    No     2909
    Yes      10
    dtype: int64




```python
pd.value_counts(data['RoofMatl'].values, sort=True)
```




    CompShg    2876
    Tar&Grv      23
    WdShake       9
    WdShngl       7
    Membran       1
    ClyTile       1
    Metal         1
    Roll          1
    dtype: int64




```python
pd.value_counts(data['SaleCondition'].values, sort=True)
```




    Normal     2402
    Partial     245
    Abnorml     190
    Family       46
    Alloca       24
    AdjLand      12
    dtype: int64




```python
pd.value_counts(data['Utilities'].values, sort=True)
```




    AllPub    2916
    NoSeWa       1
    dtype: int64




```python
data = pd.get_dummies(data, drop_first=True)

```


```python

```


```python
data= data.drop('Functional')
data['BedroomAbvGr'] = pow(data['BedroomAbvGr'], 2)
data['BsmtFullBath'] = pow(data['BsmtFullBath'], 2)
data['BsmtHalfBath'] = pow(data['BsmtHalfBath'], 2)
data['FullBath'] = pow(data['FullBath'], 2)
data['Fireplaces'] = pow(data['Fireplaces'], 2)
data['GarageCars'] = pow(data['GarageCars'], 2)
#data['BsmtFinSF2'] = pow(data['BsmtFinSF1'], 2)
#data['CentralAir'] = pow(data['CentralAir'], 2)
data['SaleType_New'] = pow(data['SaleType_New'], 2)
#data = data.drop('OverallCond')
```


```python
data.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>1stFlrSF</th>
      <th>2ndFlrSF</th>
      <th>3SsnPorch</th>
      <th>BedroomAbvGr</th>
      <th>BsmtFinSF1</th>
      <th>BsmtFinSF2</th>
      <th>BsmtFullBath</th>
      <th>BsmtHalfBath</th>
      <th>BsmtUnfSF</th>
      <th>CentralAir</th>
      <th>...</th>
      <th>SaleType_CWD</th>
      <th>SaleType_Con</th>
      <th>SaleType_ConLD</th>
      <th>SaleType_ConLI</th>
      <th>SaleType_ConLw</th>
      <th>SaleType_New</th>
      <th>SaleType_Oth</th>
      <th>SaleType_WD</th>
      <th>Street_Pave</th>
      <th>Utilities_NoSeWa</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>856</td>
      <td>854</td>
      <td>0</td>
      <td>9</td>
      <td>706.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>150.0</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1262</td>
      <td>0</td>
      <td>0</td>
      <td>9</td>
      <td>978.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>284.0</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>920</td>
      <td>866</td>
      <td>0</td>
      <td>9</td>
      <td>486.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>434.0</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>961</td>
      <td>756</td>
      <td>0</td>
      <td>9</td>
      <td>216.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>540.0</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1145</td>
      <td>1053</td>
      <td>0</td>
      <td>16</td>
      <td>655.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>490.0</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 243 columns</p>
</div>




```python
train = data.iloc[:1460]
test = data.iloc[1460:]
```


```python
train.info()
train = train.fillna(0)
train.to_csv('train data.csv')
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 1460 entries, 0 to 1459
    Columns: 243 entries, 1stFlrSF to Utilities_NoSeWa
    dtypes: float64(12), int64(26), uint8(205)
    memory usage: 737.1 KB
    


```python
test.info()
test=test.fillna(0)
test.to_csv('test_data.csv')
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 1459 entries, 0 to 1458
    Columns: 243 entries, 1stFlrSF to Utilities_NoSeWa
    dtypes: float64(12), int64(26), uint8(205)
    memory usage: 736.6 KB
    


```python

```


```python

```
