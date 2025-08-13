// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

import "@openzeppelin/contracts/proxy/ERC1967/ERC1967Proxy.sol";
import "./ArthaCoin.sol";
import "./CycleManager.sol";
import "./BurnManager.sol";
import "./AntiWhaleManager.sol";

/**
 * @title ArthaCoinDeployment
 * @notice Deployment script for complete ArthaCoin ecosystem
 * @dev Deploys and configures all contracts with proper initialization
 */
contract ArthaCoinDeployment {
    
    struct DeploymentAddresses {
        address arthaCoinProxy;
        address arthaCoinImplementation;
        address cycleManagerProxy;
        address cycleManagerImplementation;
        address burnManagerProxy;
        address burnManagerImplementation;
        address antiWhaleManagerProxy;
        address antiWhaleManagerImplementation;
    }
    
    struct PoolAddresses {
        address validatorsPool;
        address stakingRewardsPool;
        address ecosystemGrantsPool;
        address marketingWallet;
        address developersPool;
        address daoGovernancePool;
        address treasuryReserve;
    }
    
    event ArthaCoinDeployed(
        address indexed arthaCoinProxy,
        address indexed admin,
        PoolAddresses pools
    );
    
    event ManagerContractsDeployed(
        address indexed cycleManager,
        address indexed burnManager,
        address indexed antiWhaleManager
    );
    
    /**
     * @notice Deploys the complete ArthaCoin ecosystem
     * @param admin Address to receive admin roles across all contracts
     * @param pools Pool addresses for token allocations
     * @return addresses Struct containing all deployed contract addresses
     */
    function deployComplete(
        address admin,
        PoolAddresses memory pools
    ) external returns (DeploymentAddresses memory addresses) {
        
        require(admin != address(0), "Admin cannot be zero address");
        require(pools.validatorsPool != address(0), "Validators pool cannot be zero");
        require(pools.stakingRewardsPool != address(0), "Staking pool cannot be zero");
        require(pools.ecosystemGrantsPool != address(0), "Ecosystem pool cannot be zero");
        require(pools.marketingWallet != address(0), "Marketing wallet cannot be zero");
        require(pools.developersPool != address(0), "Developers pool cannot be zero");
        require(pools.daoGovernancePool != address(0), "DAO pool cannot be zero");
        require(pools.treasuryReserve != address(0), "Treasury cannot be zero");
        
        // Deploy manager contracts first
        addresses = _deployManagerContracts(admin);
        
        // Deploy ArthaCoin token contract
        addresses = _deployArthaCoin(admin, pools, addresses);
        
        // Configure manager contracts to work with token
        _configureManagerContracts(addresses);
        
        // Set manager contracts in token
        _setManagersInToken(addresses);
        
        emit ArthaCoinDeployed(addresses.arthaCoinProxy, admin, pools);
        emit ManagerContractsDeployed(
            addresses.cycleManagerProxy,
            addresses.burnManagerProxy,
            addresses.antiWhaleManagerProxy
        );
        
        return addresses;
    }
    
    /**
     * @notice Deploys manager contracts (CycleManager, BurnManager, AntiWhaleManager)
     */
    function _deployManagerContracts(address admin) internal returns (DeploymentAddresses memory addresses) {
        
        // Deploy CycleManager
        addresses.cycleManagerImplementation = address(new CycleManager());
        bytes memory cycleManagerInit = abi.encodeWithSelector(
            CycleManager.initialize.selector,
            admin,
            address(0) // Token contract address will be set later
        );
        addresses.cycleManagerProxy = address(new ERC1967Proxy(
            addresses.cycleManagerImplementation,
            cycleManagerInit
        ));
        
        // Deploy BurnManager
        addresses.burnManagerImplementation = address(new BurnManager());
        bytes memory burnManagerInit = abi.encodeWithSelector(
            BurnManager.initialize.selector,
            admin
        );
        addresses.burnManagerProxy = address(new ERC1967Proxy(
            addresses.burnManagerImplementation,
            burnManagerInit
        ));
        
        // Deploy AntiWhaleManager
        addresses.antiWhaleManagerImplementation = address(new AntiWhaleManager());
        bytes memory antiWhaleManagerInit = abi.encodeWithSelector(
            AntiWhaleManager.initialize.selector,
            admin,
            address(0) // Token contract address will be set later
        );
        addresses.antiWhaleManagerProxy = address(new ERC1967Proxy(
            addresses.antiWhaleManagerImplementation,
            antiWhaleManagerInit
        ));
        
        return addresses;
    }
    
    /**
     * @notice Deploys ArthaCoin token contract
     */
    function _deployArthaCoin(
        address admin,
        PoolAddresses memory pools,
        DeploymentAddresses memory addresses
    ) internal returns (DeploymentAddresses memory) {
        
        // Deploy ArthaCoin
        addresses.arthaCoinImplementation = address(new ArthaCoin());
        bytes memory arthaCoinInit = abi.encodeWithSelector(
            ArthaCoin.initialize.selector,
            admin,
            pools.validatorsPool,
            pools.stakingRewardsPool,
            pools.ecosystemGrantsPool,
            pools.marketingWallet,
            pools.developersPool,
            pools.daoGovernancePool,
            pools.treasuryReserve
        );
        addresses.arthaCoinProxy = address(new ERC1967Proxy(
            addresses.arthaCoinImplementation,
            arthaCoinInit
        ));
        
        return addresses;
    }
    
    /**
     * @notice Configures manager contracts to work with the token contract
     */
    function _configureManagerContracts(DeploymentAddresses memory addresses) internal {
        
        // Update CycleManager with token contract address
        CycleManager cycleManager = CycleManager(addresses.cycleManagerProxy);
        cycleManager.updateTokenContract(addresses.arthaCoinProxy);
        
        // Update AntiWhaleManager with token contract address
        AntiWhaleManager antiWhaleManager = AntiWhaleManager(addresses.antiWhaleManagerProxy);
        antiWhaleManager.updateTokenContract(addresses.arthaCoinProxy);
    }
    
    /**
     * @notice Sets manager contracts in the token contract
     */
    function _setManagersInToken(DeploymentAddresses memory addresses) internal {
        
        ArthaCoin arthaCoin = ArthaCoin(addresses.arthaCoinProxy);
        
        // Set manager contracts
        arthaCoin.setCycleManager(addresses.cycleManagerProxy);
        arthaCoin.setBurnManager(addresses.burnManagerProxy);
        arthaCoin.setAntiWhaleManager(addresses.antiWhaleManagerProxy);
    }
    
    /**
     * @notice Verifies deployment by checking basic functionality
     */
    function verifyDeployment(DeploymentAddresses memory addresses) external view returns (bool) {
        
        ArthaCoin arthaCoin = ArthaCoin(addresses.arthaCoinProxy);
        CycleManager cycleManager = CycleManager(addresses.cycleManagerProxy);
        BurnManager burnManager = BurnManager(addresses.burnManagerProxy);
        AntiWhaleManager antiWhaleManager = AntiWhaleManager(addresses.antiWhaleManagerProxy);
        
        // Check basic properties
        require(keccak256(bytes(arthaCoin.name())) == keccak256(bytes("ArthaCoin")), "Wrong token name");
        require(keccak256(bytes(arthaCoin.symbol())) == keccak256(bytes("ARTHA")), "Wrong token symbol");
        require(arthaCoin.decimals() == 18, "Wrong decimals");
        require(arthaCoin.totalSupply() == 0, "Total supply should be 0 initially");
        
        // Check manager contracts are set
        require(address(arthaCoin.cycleManager()) == addresses.cycleManagerProxy, "CycleManager not set");
        require(address(arthaCoin.burnManager()) == addresses.burnManagerProxy, "BurnManager not set");
        require(address(arthaCoin.antiWhaleManager()) == addresses.antiWhaleManagerProxy, "AntiWhaleManager not set");
        
        // Check cycle manager
        require(cycleManager.INITIAL_EMISSION() == 50_000_000 * 1e18, "Wrong initial emission");
        require(cycleManager.CYCLE_LENGTH() == 3 * 365 days, "Wrong cycle length");
        
        // Check burn manager
        require(burnManager.getBurnRateForYear(0) == 4000, "Wrong initial burn rate"); // 40%
        require(burnManager.getBurnRateForYear(16) == 9600, "Wrong final burn rate"); // 96%
        
        // Check anti-whale manager
        require(antiWhaleManager.MAX_HOLDING_PERCENTAGE() == 150, "Wrong holding percentage"); // 1.5%
        require(antiWhaleManager.MAX_TRANSFER_PERCENTAGE() == 50, "Wrong transfer percentage"); // 0.5%
        
        return true;
    }
    
    /**
     * @notice Creates example pool addresses for testing
     * @dev Only use for testing/development, not production
     */
    function createExamplePools() external pure returns (PoolAddresses memory pools) {
        pools = PoolAddresses({
            validatorsPool: address(0x1111111111111111111111111111111111111111),
            stakingRewardsPool: address(0x2222222222222222222222222222222222222222),
            ecosystemGrantsPool: address(0x3333333333333333333333333333333333333333),
            marketingWallet: address(0x4444444444444444444444444444444444444444),
            developersPool: address(0x5555555555555555555555555555555555555555),
            daoGovernancePool: address(0x6666666666666666666666666666666666666666),
            treasuryReserve: address(0x7777777777777777777777777777777777777777)
        });
    }
} 