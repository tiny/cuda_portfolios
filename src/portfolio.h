/*
*/
#pragma once

//#include <iostream>
#include <cuda_runtime.h>
#include <device_atomic_functions.h>
#include <stdint.h>
#include <string.h>

// Observable 32-bit unsigned integer for GPU
class ObservableUInt32 {
public:
    uint32_t* d_value;

    ObservableUInt32(uint32_t initialValue = 0) ;
    ~ObservableUInt32() ;
    __host__ void set(uint32_t newValue) ;
    __device__ void set_device(uint32_t newValue) ;
    __host__ __device__ uint32_t get() const ;
};

// Forward declaration
struct Holding;
struct Portfolio;

// Stock class that holds an observable price and list of observers
struct Stock {
    char symbol[8];
    ObservableUInt32 price;
    Holding** observers; // List of Holdings observing this stock
    int num_observers;

    Stock(const char* sym, uint32_t initialPrice) ;
    __host__ __device__ void addObserver(Holding* holding) ;
    __device__ void notifyObservers(uint32_t oldPrice, uint32_t newPrice) ;
};

// Holding structure, observes stock price changes and updates portfolio
struct Holding {
    char symbol[8];
    uint32_t quantity;
    uint32_t purchase_value;
    uint32_t* current_value;
    Portfolio* portfolio;
    Stock* stock;

    Holding(const char* sym, uint32_t qty, uint32_t purchaseVal, Portfolio* port, Stock* stk) ;
    __device__ void updateValue(uint32_t newPrice) ;
};

// Portfolio structure holding multiple holdings and observing total value
struct Portfolio {
    int32_t id;
    Holding** holdings;
    int num_holdings;
    ObservableUInt32 overall_value;

    Portfolio(int32_t portfolioId) ;
    __host__ void addHolding(Holding* holding) ;
    __device__ void addHolding_device(Holding* holding) ;
    __device__ void updateOverallValue_device(uint32_t oldValue, uint32_t newValue) ;
};

