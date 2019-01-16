'''
MinMaxScaler that inherits from the sklearn MinMaxScaler. They only differ in
their inverse_transform() implementation.

Created on Jan 16, 2019

@author: ameier
'''
from sklearn.preprocessing.data import MinMaxScaler


class MyMinMaxScaler(MinMaxScaler):
    '''
    Changes the inverse_transform method.
    '''

    def inverse_transform(self, X, only_range=False):
        '''
        @param X: data to be inverse transformed
        @param only_range: False for normal inverse transformation behavior
        (i.e. that of the super class). If True, only the width of the data 
        range is adapted but not the position of that range.
        '''
        if only_range:
            # e.g. for re-scaling aleatoric uncertainty: only the range should
            # be adapted but not the "position" since the values have to be
            # positive
            X = super(MyMinMaxScaler, self).inverse_transform(X)
            # In inverse_transform() are the last two lines:
            #      X -= self.min_
            #      X /= self.scale_
            # In order to only adapt the width of the range but not the position
            # only the last line X /= self.scale_ is needed. Therefore here
            # the second last line X -= self.min_ is un-done.
            X *= self.scale_  # undo
            X += self.min_   # undo
            X /= self.scale_  # redo
            return X
        else:
            return super(MyMinMaxScaler, self).inverse_transform(X)
