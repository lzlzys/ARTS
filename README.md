
## Quick-Start
ICLR_前綴是ICLR的具體代碼，PEPLER_前綴是gp2的具體代碼，通過new_main函數進行模型的訓練，process_前綴是之前進行數據處理寫的。
data文件夾中是需要加載的數據，prompt_generator 是需要加載的提示文件。
If you want totrain the amazon dataset, you can train the model just with data_name input（in our model we choose three amazon dataset:Tools_and_Home_Improvement, Sports_and_Outdoors, Toys_and_Games).
```
python new_main.py --amazon=True --data_name=[data_name]
```
If you want to train the yelp dataset, you can train the model just with data_name input（in our model we choose two yelp dataset:tampa, new orleans).
```
python new_main.py --amazon=False --data_name=[data_name]
```
If you want to change the parameters, just set the additional command parameters as you need. For example:
```
python new_main.py --data_name=tampa --num_hidden_layers=4 --batch_size=10
```

You can also test the model has been saved by command line.
```
python new_main.py --data_name=tampa --do_eval=True
```



