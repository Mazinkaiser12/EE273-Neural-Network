#pragma once
#include "Payment.h"
#include <vector>
#include <string>

class Payment;

class Account {

public:
	Account(std::string name, double balance) : name(name), balance(balance) {}
	void addPaymentMethod(Payment* paymentMethod);
	void makePayment(double amount);
	double getBalance();
	bool deductFromBalance(double amount);
private:
	double balance;
	std::string name;
	std::vector<Payment*> paymentMethods;
};