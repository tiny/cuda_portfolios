/*
*/
#include "portfolio.h"

//#if __CUDA_ARCH__ < 200
//#  pragma message( "CUDA_ARCH not defined" )
//#endif

//-----------------------------------------------------------------------------
ObservableUInt32::ObservableUInt32(uint32_t initialValue ) {
    cudaMallocManaged(&d_value, sizeof(uint32_t));  // CUDA-managed memory
    *d_value = initialValue;
}

ObservableUInt32::~ObservableUInt32() {
    cudaFree(d_value);
}

__host__ void ObservableUInt32::set(uint32_t newValue) {
    cudaMemcpy(d_value, &newValue, sizeof(uint32_t), cudaMemcpyHostToDevice);
}

__device__ void ObservableUInt32::set_device(uint32_t newValue) {
    atomicExch(d_value, newValue);
}

__host__ __device__ uint32_t ObservableUInt32::get() const {
    return *d_value;
}

//-----------------------------------------------------------------------------
Stock::Stock(const char* sym, uint32_t initialPrice) : price(initialPrice), num_observers(0) {
    strcpy(symbol, sym);
    cudaMallocManaged(&observers, 100 * sizeof(Holding*));  // Arbitrary limit of 100 observers per stock
}

__host__ __device__ void Stock::addObserver(Holding* holding) {
    observers[num_observers++] = holding;
}

__device__ void Stock::notifyObservers(uint32_t oldPrice, uint32_t newPrice) {
    for (int i = 0; i < num_observers; i++) {
        observers[i]->updateValue(newPrice);
    }
}

//-----------------------------------------------------------------------------
Holding::Holding(const char* sym, uint32_t qty, uint32_t purchaseVal, Portfolio* port, Stock* stk)
: quantity(qty), purchase_value(purchaseVal), portfolio(port), stock(stk) {
  strcpy(symbol, sym);
  cudaMallocManaged(&current_value, sizeof(uint32_t));
  *current_value = qty * stk->price.get();
  stk->addObserver(this);
}

__device__ void Holding::updateValue(uint32_t newPrice) {
    uint32_t oldValue = *current_value;
    *current_value = quantity * newPrice;
    portfolio->updateOverallValue_device(oldValue, *current_value);
}

//-----------------------------------------------------------------------------
Portfolio::Portfolio(int32_t portfolioId) : id(portfolioId), num_holdings(0), overall_value(0) {
    cudaMallocManaged(&holdings, 100 * sizeof(Holding*));  // Arbitrary limit of 100 holdings per portfolio
}

__host__ void Portfolio::addHolding(Holding* holding) {
    holdings[num_holdings++] = holding;
    uint32_t currentValue = *(holding->current_value);
    cudaMemcpy(overall_value.d_value, &currentValue, sizeof(uint32_t), cudaMemcpyHostToDevice);
}

__device__ void Portfolio::addHolding_device(Holding* holding) {
    holdings[num_holdings++] = holding;
    atomicAdd(overall_value.d_value, *(holding->current_value));
}

//    __host__ void updateOverallValue(uint32_t oldValue, uint32_t newValue) {
//        uint32_t updatedValue = get() - oldValue + newValue;
//        cudaMemcpy(overall_value.d_value, &updatedValue, sizeof(uint32_t), cudaMemcpyHostToDevice);
//    }

__device__ void Portfolio::updateOverallValue_device(uint32_t oldValue, uint32_t newValue) {
    atomicSub(overall_value.d_value, oldValue);
    atomicAdd(overall_value.d_value, newValue);
}

//-----------------------------------------------------------------------------
