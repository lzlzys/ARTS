
## Quick-Start

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



