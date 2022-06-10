from AWNN import AWNN
import numpy as np
from sklearn.model_selection import GridSearchCV




dim=8
n_train=300
n_test=300
distribution=1
np.random.seed(2)
density=TestDistribution(distribution,dim).returnDistribution()

X_train, pdf_X_train = density.generate(n_train)

X_test, pdf_X_test = density.generate(n_test)

nn_model=AWNN()

nn_model.fit(X_train)


parameters={"C":[1,2,3,4,5,6]}
clf=GridSearchCV(estimator=nn_model,param_grid=parameters)
clf.fit(X_test)
#np.exp(nn_model.log_density)
#np.exp(nn_model.log_density)/pdf_X_train
#nn_model.predict(X_test)

#np.exp(nn_model.log_density)/pdf_X_test