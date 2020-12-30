#include<bits/stdc++.h>

using namespace std;

const double eps = 1e-8;
const double alpha = 0.01;

int m;
double sumx, sumy, sumxy, sumx2;
double th0, th1;

int main(){
	freopen("ex1data1.txt", "r", stdin);
	double x, y;
	while(scanf("%lf,%lf", &x, &y) != EOF)
		m++, sumx += x, sumy += y, sumx2 += x * x, sumxy += x * y;
	th0 = th1 = 0;
	while(1){
		double nth0, nth1;
		nth0 = th0 - alpha * 1.0 / m * (th1 * sumx - sumy + m * th0);
		nth1 = th1 - alpha * 1.0 / m * (th1 * sumx2 - sumxy + sumx * th0);
		if(fabs(th0 - nth0) < eps && fabs(th1 - nth1) < eps)	break;
		th0 = nth0, th1 = nth1;
	}
	printf("%f %f\n", th0, th1);
	return 0;
}