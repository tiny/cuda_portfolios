#include <iostream>
#include <map>
#include <vector>
#include <cuda_runtime.h>
#include <device_atomic_functions.h>
#include <stdint.h>
#include <string.h>

// Observable 32-bit unsigned integer for GPU
class ObservableUInt32 {
public:
    uint32_t* d_value;

    ObservableUInt32(uint32_t initialValue = 0) {
        cudaMallocManaged(&d_value, sizeof(uint32_t));  // CUDA-managed memory
        *d_value = initialValue;
    }

    ~ObservableUInt32() {
        cudaFree(d_value);
    }

    __device__ void set(uint32_t newValue) {
        atomicExch(d_value, newValue);
    }

    __device__ uint32_t get() const {
        return *d_value;
    }
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

    Stock(const char* sym, uint32_t initialPrice) : price(initialPrice), num_observers(0) {
        strcpy(symbol, sym);
        cudaMallocManaged(&observers, 100 * sizeof(Holding*));  // Arbitrary limit of 100 observers per stock
    }

    __device__ void addObserver(Holding* holding) {
        observers[num_observers++] = holding;
    }

    __device__ void notifyObservers(uint32_t oldPrice, uint32_t newPrice);
};

// Holding structure, observes stock price changes and updates portfolio
struct Holding {
    char symbol[8];
    uint32_t quantity;
    uint32_t purchase_value;
    uint32_t* current_value;
    Portfolio* portfolio;
    Stock* stock;

    Holding(const char* sym, uint32_t qty, uint32_t purchaseVal, Portfolio* port, Stock* stk)
        : quantity(qty), purchase_value(purchaseVal), portfolio(port), stock(stk) {
        strcpy(symbol, sym);
        cudaMallocManaged(&current_value, sizeof(uint32_t));
        *current_value = qty * stk->price.get();
        stk->addObserver(this);
    }

    __device__ void updateValue(uint32_t newPrice);
};

// Portfolio structure holding multiple holdings and observing total value
struct Portfolio {
    int32_t id;
    Holding** holdings;
    int num_holdings;
    ObservableUInt32 overall_value;

    Portfolio(int32_t portfolioId) : id(portfolioId), num_holdings(0), overall_value(0) {
        cudaMallocManaged(&holdings, 100 * sizeof(Holding*));  // Arbitrary limit of 100 holdings per portfolio
    }

    __device__ void addHolding(Holding* holding) {
        holdings[num_holdings++] = holding;
        atomicAdd(overall_value.d_value, *(holding->current_value));
    }

    __device__ void updateOverallValue(uint32_t oldValue, uint32_t newValue) {
        atomicSub(overall_value.d_value, oldValue);
        atomicAdd(overall_value.d_value, newValue);
    }
};

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
    portfolio->updateOverallValue(oldValue, *current_value);
}

// GPU Kernel to update stock price and notify observers
__global__ void updateStockPriceKernel(Stock* stock, uint32_t newPrice) {
    uint32_t oldPrice = stock->price.get();
    stock->price.set(newPrice);
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
    for (const auto& [id, portfolio] : portfolios) {
        std::cout << "Portfolio " << id << " new overall value: $" 
                  << portfolio->overall_value.get() << std::endl;
    }

    return 0;
}
