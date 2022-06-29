{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "123ea2ad",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "dict_keys(['data', 'target', 'frame', 'categories', 'feature_names', 'target_names', 'DESCR', 'details', 'url'])"
      ]
     },
     "execution_count": 4,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "from sklearn.datasets import fetch_openml\n",
    "mnist = fetch_openml('mnist_784', version = 1)\n",
    "mnist.keys()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "id": "a77b6059",
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "import matplotlib as mpt\n",
    "import matplotlib.pyplot as plt\n",
    "\n",
    "x, y = mnist['data'], mnist['target']\n",
    "#change targets from str to int\n",
    "y = y.astype(np.uint8)\n",
    "x_train, x_test, y_train, y_test = x[:60000], x[60000:], y[:60000], y[60000:]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 48,
   "id": "c2db69a2",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "SGDClassifier(random_state=42)"
      ]
     },
     "execution_count": 48,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#make a 5 identicator\n",
    "y_train_5 = (y_train == 5)\n",
    "y_test_5 = (y_test == 5)\n",
    "\n",
    "from sklearn.linear_model import SGDClassifier\n",
    "\n",
    "sgd_clf = SGDClassifier(random_state=42)\n",
    "sgd_clf.fit(x_train, y_train_5)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 64,
   "id": "3a12aeb6",
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "ename": "KeyError",
     "evalue": "1",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mKeyError\u001b[0m                                  Traceback (most recent call last)",
      "File \u001b[1;32m~\\AppData\\Local\\Programs\\Python\\Python310\\lib\\site-packages\\pandas\\core\\indexes\\base.py:3621\u001b[0m, in \u001b[0;36mIndex.get_loc\u001b[1;34m(self, key, method, tolerance)\u001b[0m\n\u001b[0;32m   3620\u001b[0m \u001b[38;5;28;01mtry\u001b[39;00m:\n\u001b[1;32m-> 3621\u001b[0m     \u001b[38;5;28;01mreturn\u001b[39;00m \u001b[38;5;28;43mself\u001b[39;49m\u001b[38;5;241;43m.\u001b[39;49m\u001b[43m_engine\u001b[49m\u001b[38;5;241;43m.\u001b[39;49m\u001b[43mget_loc\u001b[49m\u001b[43m(\u001b[49m\u001b[43mcasted_key\u001b[49m\u001b[43m)\u001b[49m\n\u001b[0;32m   3622\u001b[0m \u001b[38;5;28;01mexcept\u001b[39;00m \u001b[38;5;167;01mKeyError\u001b[39;00m \u001b[38;5;28;01mas\u001b[39;00m err:\n",
      "File \u001b[1;32m~\\AppData\\Local\\Programs\\Python\\Python310\\lib\\site-packages\\pandas\\_libs\\index.pyx:136\u001b[0m, in \u001b[0;36mpandas._libs.index.IndexEngine.get_loc\u001b[1;34m()\u001b[0m\n",
      "File \u001b[1;32m~\\AppData\\Local\\Programs\\Python\\Python310\\lib\\site-packages\\pandas\\_libs\\index.pyx:163\u001b[0m, in \u001b[0;36mpandas._libs.index.IndexEngine.get_loc\u001b[1;34m()\u001b[0m\n",
      "File \u001b[1;32mpandas\\_libs\\hashtable_class_helper.pxi:5198\u001b[0m, in \u001b[0;36mpandas._libs.hashtable.PyObjectHashTable.get_item\u001b[1;34m()\u001b[0m\n",
      "File \u001b[1;32mpandas\\_libs\\hashtable_class_helper.pxi:5206\u001b[0m, in \u001b[0;36mpandas._libs.hashtable.PyObjectHashTable.get_item\u001b[1;34m()\u001b[0m\n",
      "\u001b[1;31mKeyError\u001b[0m: 1",
      "\nThe above exception was the direct cause of the following exception:\n",
      "\u001b[1;31mKeyError\u001b[0m                                  Traceback (most recent call last)",
      "Input \u001b[1;32mIn [64]\u001b[0m, in \u001b[0;36m<module>\u001b[1;34m\u001b[0m\n\u001b[1;32m----> 1\u001b[0m \u001b[38;5;28mprint\u001b[39m(\u001b[43mx_train\u001b[49m\u001b[43m[\u001b[49m\u001b[38;5;241;43m1\u001b[39;49m\u001b[43m]\u001b[49m[\u001b[38;5;241m1\u001b[39m])\n",
      "File \u001b[1;32m~\\AppData\\Local\\Programs\\Python\\Python310\\lib\\site-packages\\pandas\\core\\frame.py:3505\u001b[0m, in \u001b[0;36mDataFrame.__getitem__\u001b[1;34m(self, key)\u001b[0m\n\u001b[0;32m   3503\u001b[0m \u001b[38;5;28;01mif\u001b[39;00m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39mcolumns\u001b[38;5;241m.\u001b[39mnlevels \u001b[38;5;241m>\u001b[39m \u001b[38;5;241m1\u001b[39m:\n\u001b[0;32m   3504\u001b[0m     \u001b[38;5;28;01mreturn\u001b[39;00m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39m_getitem_multilevel(key)\n\u001b[1;32m-> 3505\u001b[0m indexer \u001b[38;5;241m=\u001b[39m \u001b[38;5;28;43mself\u001b[39;49m\u001b[38;5;241;43m.\u001b[39;49m\u001b[43mcolumns\u001b[49m\u001b[38;5;241;43m.\u001b[39;49m\u001b[43mget_loc\u001b[49m\u001b[43m(\u001b[49m\u001b[43mkey\u001b[49m\u001b[43m)\u001b[49m\n\u001b[0;32m   3506\u001b[0m \u001b[38;5;28;01mif\u001b[39;00m is_integer(indexer):\n\u001b[0;32m   3507\u001b[0m     indexer \u001b[38;5;241m=\u001b[39m [indexer]\n",
      "File \u001b[1;32m~\\AppData\\Local\\Programs\\Python\\Python310\\lib\\site-packages\\pandas\\core\\indexes\\base.py:3623\u001b[0m, in \u001b[0;36mIndex.get_loc\u001b[1;34m(self, key, method, tolerance)\u001b[0m\n\u001b[0;32m   3621\u001b[0m     \u001b[38;5;28;01mreturn\u001b[39;00m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39m_engine\u001b[38;5;241m.\u001b[39mget_loc(casted_key)\n\u001b[0;32m   3622\u001b[0m \u001b[38;5;28;01mexcept\u001b[39;00m \u001b[38;5;167;01mKeyError\u001b[39;00m \u001b[38;5;28;01mas\u001b[39;00m err:\n\u001b[1;32m-> 3623\u001b[0m     \u001b[38;5;28;01mraise\u001b[39;00m \u001b[38;5;167;01mKeyError\u001b[39;00m(key) \u001b[38;5;28;01mfrom\u001b[39;00m \u001b[38;5;21;01merr\u001b[39;00m\n\u001b[0;32m   3624\u001b[0m \u001b[38;5;28;01mexcept\u001b[39;00m \u001b[38;5;167;01mTypeError\u001b[39;00m:\n\u001b[0;32m   3625\u001b[0m     \u001b[38;5;66;03m# If we have a listlike key, _check_indexing_error will raise\u001b[39;00m\n\u001b[0;32m   3626\u001b[0m     \u001b[38;5;66;03m#  InvalidIndexError. Otherwise we fall through and re-raise\u001b[39;00m\n\u001b[0;32m   3627\u001b[0m     \u001b[38;5;66;03m#  the TypeError.\u001b[39;00m\n\u001b[0;32m   3628\u001b[0m     \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39m_check_indexing_error(key)\n",
      "\u001b[1;31mKeyError\u001b[0m: 1"
     ]
    }
   ],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": 57,
   "id": "be03c680",
   "metadata": {},
   "outputs": [
    {
     "ename": "KeyError",
     "evalue": "1",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mKeyError\u001b[0m                                  Traceback (most recent call last)",
      "File \u001b[1;32m~\\AppData\\Local\\Programs\\Python\\Python310\\lib\\site-packages\\pandas\\core\\indexes\\base.py:3621\u001b[0m, in \u001b[0;36mIndex.get_loc\u001b[1;34m(self, key, method, tolerance)\u001b[0m\n\u001b[0;32m   3620\u001b[0m \u001b[38;5;28;01mtry\u001b[39;00m:\n\u001b[1;32m-> 3621\u001b[0m     \u001b[38;5;28;01mreturn\u001b[39;00m \u001b[38;5;28;43mself\u001b[39;49m\u001b[38;5;241;43m.\u001b[39;49m\u001b[43m_engine\u001b[49m\u001b[38;5;241;43m.\u001b[39;49m\u001b[43mget_loc\u001b[49m\u001b[43m(\u001b[49m\u001b[43mcasted_key\u001b[49m\u001b[43m)\u001b[49m\n\u001b[0;32m   3622\u001b[0m \u001b[38;5;28;01mexcept\u001b[39;00m \u001b[38;5;167;01mKeyError\u001b[39;00m \u001b[38;5;28;01mas\u001b[39;00m err:\n",
      "File \u001b[1;32m~\\AppData\\Local\\Programs\\Python\\Python310\\lib\\site-packages\\pandas\\_libs\\index.pyx:136\u001b[0m, in \u001b[0;36mpandas._libs.index.IndexEngine.get_loc\u001b[1;34m()\u001b[0m\n",
      "File \u001b[1;32m~\\AppData\\Local\\Programs\\Python\\Python310\\lib\\site-packages\\pandas\\_libs\\index.pyx:163\u001b[0m, in \u001b[0;36mpandas._libs.index.IndexEngine.get_loc\u001b[1;34m()\u001b[0m\n",
      "File \u001b[1;32mpandas\\_libs\\hashtable_class_helper.pxi:5198\u001b[0m, in \u001b[0;36mpandas._libs.hashtable.PyObjectHashTable.get_item\u001b[1;34m()\u001b[0m\n",
      "File \u001b[1;32mpandas\\_libs\\hashtable_class_helper.pxi:5206\u001b[0m, in \u001b[0;36mpandas._libs.hashtable.PyObjectHashTable.get_item\u001b[1;34m()\u001b[0m\n",
      "\u001b[1;31mKeyError\u001b[0m: 1",
      "\nThe above exception was the direct cause of the following exception:\n",
      "\u001b[1;31mKeyError\u001b[0m                                  Traceback (most recent call last)",
      "Input \u001b[1;32mIn [57]\u001b[0m, in \u001b[0;36m<module>\u001b[1;34m\u001b[0m\n\u001b[0;32m      1\u001b[0m \u001b[38;5;28;01mfor\u001b[39;00m train_index, test_index \u001b[38;5;129;01min\u001b[39;00m skfolds\u001b[38;5;241m.\u001b[39msplit(x_train, y_train_5):\n\u001b[0;32m      2\u001b[0m     clone_clf \u001b[38;5;241m=\u001b[39m clone(sgd_clf)\n\u001b[1;32m----> 3\u001b[0m     x_train_folds \u001b[38;5;241m=\u001b[39m \u001b[43mx_train\u001b[49m\u001b[43m[\u001b[49m\u001b[43mtrain_index\u001b[49m\u001b[43m[\u001b[49m\u001b[38;5;241;43m0\u001b[39;49m\u001b[43m]\u001b[49m\u001b[43m]\u001b[49m\n\u001b[0;32m      4\u001b[0m     y_train_folds \u001b[38;5;241m=\u001b[39m y_train_5[train_index]\n\u001b[0;32m      5\u001b[0m     x_test_fold \u001b[38;5;241m=\u001b[39m x_train[test_index]\n",
      "File \u001b[1;32m~\\AppData\\Local\\Programs\\Python\\Python310\\lib\\site-packages\\pandas\\core\\frame.py:3505\u001b[0m, in \u001b[0;36mDataFrame.__getitem__\u001b[1;34m(self, key)\u001b[0m\n\u001b[0;32m   3503\u001b[0m \u001b[38;5;28;01mif\u001b[39;00m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39mcolumns\u001b[38;5;241m.\u001b[39mnlevels \u001b[38;5;241m>\u001b[39m \u001b[38;5;241m1\u001b[39m:\n\u001b[0;32m   3504\u001b[0m     \u001b[38;5;28;01mreturn\u001b[39;00m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39m_getitem_multilevel(key)\n\u001b[1;32m-> 3505\u001b[0m indexer \u001b[38;5;241m=\u001b[39m \u001b[38;5;28;43mself\u001b[39;49m\u001b[38;5;241;43m.\u001b[39;49m\u001b[43mcolumns\u001b[49m\u001b[38;5;241;43m.\u001b[39;49m\u001b[43mget_loc\u001b[49m\u001b[43m(\u001b[49m\u001b[43mkey\u001b[49m\u001b[43m)\u001b[49m\n\u001b[0;32m   3506\u001b[0m \u001b[38;5;28;01mif\u001b[39;00m is_integer(indexer):\n\u001b[0;32m   3507\u001b[0m     indexer \u001b[38;5;241m=\u001b[39m [indexer]\n",
      "File \u001b[1;32m~\\AppData\\Local\\Programs\\Python\\Python310\\lib\\site-packages\\pandas\\core\\indexes\\base.py:3623\u001b[0m, in \u001b[0;36mIndex.get_loc\u001b[1;34m(self, key, method, tolerance)\u001b[0m\n\u001b[0;32m   3621\u001b[0m     \u001b[38;5;28;01mreturn\u001b[39;00m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39m_engine\u001b[38;5;241m.\u001b[39mget_loc(casted_key)\n\u001b[0;32m   3622\u001b[0m \u001b[38;5;28;01mexcept\u001b[39;00m \u001b[38;5;167;01mKeyError\u001b[39;00m \u001b[38;5;28;01mas\u001b[39;00m err:\n\u001b[1;32m-> 3623\u001b[0m     \u001b[38;5;28;01mraise\u001b[39;00m \u001b[38;5;167;01mKeyError\u001b[39;00m(key) \u001b[38;5;28;01mfrom\u001b[39;00m \u001b[38;5;21;01merr\u001b[39;00m\n\u001b[0;32m   3624\u001b[0m \u001b[38;5;28;01mexcept\u001b[39;00m \u001b[38;5;167;01mTypeError\u001b[39;00m:\n\u001b[0;32m   3625\u001b[0m     \u001b[38;5;66;03m# If we have a listlike key, _check_indexing_error will raise\u001b[39;00m\n\u001b[0;32m   3626\u001b[0m     \u001b[38;5;66;03m#  InvalidIndexError. Otherwise we fall through and re-raise\u001b[39;00m\n\u001b[0;32m   3627\u001b[0m     \u001b[38;5;66;03m#  the TypeError.\u001b[39;00m\n\u001b[0;32m   3628\u001b[0m     \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39m_check_indexing_error(key)\n",
      "\u001b[1;31mKeyError\u001b[0m: 1"
     ]
    }
   ],
   "source": [
    "for train_index, test_index in skfolds.split(x_train, y_train_5):\n",
    "    clone_clf = clone(sgd_clf)\n",
    "    x_train_folds = x_train[train_index]\n",
    "    y_train_folds = y_train_5[train_index]\n",
    "    x_test_fold = x_train[test_index]\n",
    "    y_test_fold = y_train_5[test_index]\n",
    "    \n",
    "    clone_clf.fit(x_train_folds, y_train_folds)\n",
    "    y_pred = clone_clf.predict(x_test_fold)\n",
    "    n_correct = sum(y_pred == y_test_fold)\n",
    "    print(n_correct / len(y_pred))"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.10.2"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
