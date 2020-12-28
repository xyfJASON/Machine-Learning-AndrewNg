#include<bits/stdc++.h>

using namespace std;

const int N = 105;
const double eps = 1e-8;
double alpha = 100;

int m;
double x[N], y[N];
double th0, th1;

inline double gradJ(int k){
	double res = 0;
	for(int i = 1; i <= m; i++)
		res += (th1 * x[i] + th0 - y[i]) * (k == 1 ? x[i] : 1);
	return res / m;
}

inline double calc(){
	double res = 0;
	for(int i = 1; i <= m; i++)
		res += (th1 * x[i] + th0 - y[i]) * (th1 * x[i] + th0 - y[i]);
	return res / m / 2;
}

int main(){
	freopen("ex1data1.txt", "r", stdin);
	while(scanf("%lf,%lf", &x[0], &y[0]) != EOF)
		m++, x[m] = x[0], y[m] = y[0];
	th0 = th1 = 0;
	double preJ = 1e9, J = 0;
	while(1){
		double nth0, nth1;
		nth0 = th0 - alpha * gradJ(0);
		nth1 = th1 - alpha * gradJ(1);
		if(fabs(th0 - nth0) < eps && fabs(th1 - nth1) < eps)	break;
		th0 = nth0, th1 = nth1;
		J = calc();
		if(J > preJ)	alpha /= 5;
		preJ = J;
	}
	printf("%f\n%f %f\n", alpha, th0, th1);
	return 0;
}