#include <iostream>
#include <map>
#include <vector>
#include "portfolio.h"

#ifdef REMOVE
// Function to update all holdings when a stock price changes
__device__ void Stock::notifyObservers(uint32_t oldPrice, uint32_t newPrice) {
    for (int i = 0; i < num_observers; i++) {
        observers[i]->updateValue(newPrice);
    }
}

// Function to update a holding's value
__device__ void Holding::updateValue(uint32_t newPrice) {
    uint32_t oldValue = *current_value;
    *current_value = quantity * newPrice;
    portfolio->updateOverallValue_device(oldValue, *current_value);
}
#endif

// GPU Kernel to update stock price and notify observers
__global__ void updateStockPriceKernel(Stock* stock, uint32_t newPrice) {
    uint32_t oldPrice = stock->price.get();
    stock->price.set_device(newPrice);
    stock->notifyObservers(oldPrice, newPrice);
}

// Function to update stock price
void updateStockPriceOnGPU(Stock* stock, uint32_t newPrice) {
    updateStockPriceKernel<<<1, 1>>>(stock, newPrice);
    cudaDeviceSynchronize();
}

// Global maps for stock and portfolio management (CPU-side)
std::map<std::string, Stock*> stockMarket;
std::map<int32_t, Portfolio*> portfolios;

// Function to add a stock
void addStock(const std::string& name, uint32_t price) {
    stockMarket[name] = new Stock(name.c_str(), price);
}

// Function to add a portfolio
void addPortfolio(int32_t accountId) {
    portfolios[accountId] = new Portfolio(accountId);
}

// Function to add a holding to a portfolio
void addHoldingToPortfolio(int32_t accountId, const std::string& stockSymbol, uint32_t quantity) {
    Portfolio* portfolio = portfolios[accountId];
    Stock* stock = stockMarket[stockSymbol];

    Holding* holding = new Holding(stockSymbol.c_str(), quantity, quantity * stock->price.get(), portfolio, stock);
    portfolio->addHolding(holding);
}

// Main function
int main() {
    // Initialize stocks
    addStock("AAPL", 150);
    addStock("GOOGL", 2800);

    // Create portfolios
    addPortfolio(101);
    addPortfolio(102);

    // Add holdings to portfolios
    addHoldingToPortfolio(101, "AAPL", 10);
    addHoldingToPortfolio(101, "GOOGL", 5);
    addHoldingToPortfolio(102, "AAPL", 20);

    // Update stock prices in GPU
    updateStockPriceOnGPU(stockMarket["AAPL"], 155);
    updateStockPriceOnGPU(stockMarket["GOOGL"], 2900);

    // Print updated portfolio values
    for (const auto& it : portfolios) {
        Portfolio *p = it.second ;
        std::cout << "Portfolio " << it.first << " new overall value: $" 
                  << p->overall_value.get() << std::endl;
    }

    return 0;
}
