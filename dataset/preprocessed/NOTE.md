# Perbedaan Hasil Data Preprocess

Hasil preprocess dari notebook dan pipeline akan berbeda karena:

1. Notebook: ... -> encode -> split -> scale
2. Pipeline: ... -> split -> encode -> scale

Perbedaan ini juga disebabkan karena:

1. `train_test_split` akan menghasilkan urutan data latih dan tes yang berbeda tergantung dari
   `random_state` dan isi datanya, yang mana di notebook encode terlebih dahulu baru split
   sedangkan di pipeline split dahulu baru encode.
2. Karena urutan datanya berubah maka `StandardScaler` juga akan menghasilkan scaling yang berbeda
   karena itu tergantung dari data yang diberikan di fit. Ini karena scaler tersebut bekerja dengan
   persamaan:
   $$
   z = {(x - \bar{u}) \over s}
   $$
   yang mana $\bar{u}$ dan $s$ adalah mean dan standar deviasi dari data yang diberikan di fit.