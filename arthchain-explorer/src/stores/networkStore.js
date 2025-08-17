import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import { NETWORKS, DEFAULT_NETWORK, LOCAL_STORAGE_KEYS } from '../constants';

const useNetworkStore = create(
  persist(
    (set, get) => ({
      // Current selected network
      selectedNetwork: DEFAULT_NETWORK,
      
      // Available networks
      networks: Object.values(NETWORKS),
      
      // Network connection status
      isConnected: false,
      isConnecting: false,
      connectionError: null,
      
      // Actions
      setSelectedNetwork: (network) => {
        set({ 
          selectedNetwork: network,
          isConnected: false,
          connectionError: null
        });
      },
      
      setConnectionStatus: (status) => {
        set({ isConnected: status });
      },
      
      setConnecting: (connecting) => {
        set({ isConnecting: connecting });
      },
      
      setConnectionError: (error) => {
        set({ 
          connectionError: error,
          isConnected: false,
          isConnecting: false
        });
      },
      
      // Get network by ID
      getNetworkById: (id) => {
        return get().networks.find(network => network.id === id);
      },
      
      // Reset connection state
      resetConnection: () => {
        set({
          isConnected: false,
          isConnecting: false,
          connectionError: null
        });
      }
    }),
    {
      name: LOCAL_STORAGE_KEYS.SELECTED_NETWORK,
      partialize: (state) => ({ selectedNetwork: state.selectedNetwork })
    }
  )
);

export default useNetworkStore;

