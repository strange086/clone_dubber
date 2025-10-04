import React from 'react'
import { motion } from 'framer-motion'

const SystemStatus = ({ systemInfo }) => {
  if (!systemInfo) {
    return (
      <div className="flex justify-center items-center h-64">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
      </div>
    )
  }

  const formatBytes = (bytes) => {
    if (!bytes) return '0 B'
    const k = 1024
    const sizes = ['B', 'KB', 'MB', 'GB', 'TB']
    const i = Math.floor(Math.log(bytes) / Math.log(k))
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i]
  }

  const getStatusColor = (status) => {
    return status ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800'
  }

  const getStatusIcon = (status) => {
    return status ? '✅' : '❌'
  }

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-2xl font-bold text-gray-900">System Status</h2>
        <p className="text-gray-600 mt-1">
          Monitor system health and AI model status
        </p>
      </div>

      {/* System Information */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="bg-white rounded-lg shadow-sm border p-6"
        >
          <h3 className="text-lg font-semibold text-gray-900 mb-4">
            💻 System Information
          </h3>
          
          <div className="space-y-3">
            <div className="flex justify-between">
              <span className="text-gray-600">Platform:</span>
              <span className="font-medium">{systemInfo.system?.platform || 'Unknown'}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-600">Python Version:</span>
              <span className="font-medium">{systemInfo.system?.python_version || 'Unknown'}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-600">CPU Cores:</span>
              <span className="font-medium">{systemInfo.system?.cpu_count || 'Unknown'}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-600">Total Memory:</span>
              <span className="font-medium">{formatBytes(systemInfo.system?.memory_total)}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-600">Available Memory:</span>
              <span className="font-medium">{formatBytes(systemInfo.system?.memory_available)}</span>
            </div>
          </div>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 }}
          className="bg-white rounded-lg shadow-sm border p-6"
        >
          <h3 className="text-lg font-semibold text-gray-900 mb-4">
            🚀 GPU Information
          </h3>
          
          <div className="space-y-3">
            <div className="flex justify-between items-center">
              <span className="text-gray-600">CUDA Available:</span>
              <span className={`px-2 py-1 text-xs font-medium rounded-full ${getStatusColor(systemInfo.system?.cuda_available)}`}>
                {getStatusIcon(systemInfo.system?.cuda_available)} {systemInfo.system?.cuda_available ? 'Yes' : 'No'}
              </span>
            </div>
            
            {systemInfo.system?.cuda_available && (
              <>
                <div className="flex justify-between">
                  <span className="text-gray-600">CUDA Version:</span>
                  <span className="font-medium">{systemInfo.system?.cuda_version || 'Unknown'}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">GPU Count:</span>
                  <span className="font-medium">{systemInfo.system?.gpu_count || 0}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">GPU Name:</span>
                  <span className="font-medium text-sm">{systemInfo.system?.gpu_name || 'Unknown'}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">GPU Memory:</span>
                  <span className="font-medium">{formatBytes(systemInfo.system?.gpu_memory)}</span>
                </div>
              </>
            )}
          </div>
        </motion.div>
      </div>

      {/* AI Models Status */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.2 }}
        className="bg-white rounded-lg shadow-sm border p-6"
      >
        <h3 className="text-lg font-semibold text-gray-900 mb-4">
          🤖 AI Models Status
        </h3>
        
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          {Object.entries(systemInfo.models || {}).map(([modelName, status]) => (
            <div key={modelName} className="text-center p-4 border rounded-lg">
              <div className="text-2xl mb-2">
                {getStatusIcon(status)}
              </div>
              <h4 className="font-medium text-gray-900 mb-1 capitalize">
                {modelName.replace(/_/g, ' ')}
              </h4>
              <span className={`px-2 py-1 text-xs font-medium rounded-full ${getStatusColor(status)}`}>
                {status ? 'Loaded' : 'Not Loaded'}
              </span>
            </div>
          ))}
        </div>
      </motion.div>

      {/* Language Support */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.3 }}
        className="bg-white rounded-lg shadow-sm border p-6"
      >
        <h3 className="text-lg font-semibold text-gray-900 mb-4">
          🌍 Supported Languages
        </h3>
        
        <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-3">
          {Object.entries(systemInfo.supported_languages || {}).map(([code, name]) => (
            <div key={code} className="flex items-center space-x-2 p-2 bg-gray-50 rounded-md">
              <span className="text-sm font-medium text-gray-900">{code.toUpperCase()}</span>
              <span className="text-sm text-gray-600">{name}</span>
            </div>
          ))}
        </div>
      </motion.div>

      {/* Voice Profiles Stats */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.4 }}
        className="bg-white rounded-lg shadow-sm border p-6"
      >
        <h3 className="text-lg font-semibold text-gray-900 mb-4">
          🎙️ Voice Cloning Statistics
        </h3>
        
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <div className="text-center">
            <div className="text-3xl font-bold text-blue-600 mb-1">
              {systemInfo.voice_profiles_count || 0}
            </div>
            <div className="text-sm text-gray-600">Voice Profiles Created</div>
          </div>
          
          <div className="text-center">
            <div className="text-3xl font-bold text-green-600 mb-1">
              {Object.values(systemInfo.models || {}).filter(Boolean).length}
            </div>
            <div className="text-sm text-gray-600">Models Loaded</div>
          </div>
          
          <div className="text-center">
            <div className="text-3xl font-bold text-purple-600 mb-1">
              {Object.keys(systemInfo.supported_languages || {}).length}
            </div>
            <div className="text-sm text-gray-600">Languages Supported</div>
          </div>
        </div>
      </motion.div>

      {/* Performance Metrics */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.5 }}
        className="bg-white rounded-lg shadow-sm border p-6"
      >
        <h3 className="text-lg font-semibold text-gray-900 mb-4">
          📊 Performance Targets (PRD)
        </h3>
        
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          <div className="text-center p-4 bg-blue-50 rounded-lg">
            <div className="text-2xl font-bold text-blue-600 mb-1">≥88%</div>
            <div className="text-sm text-gray-600">ASR Accuracy Target</div>
          </div>
          
          <div className="text-center p-4 bg-green-50 rounded-lg">
            <div className="text-2xl font-bold text-green-600 mb-1">>42</div>
            <div className="text-sm text-gray-600">Translation BLEU Score</div>
          </div>
          
          <div className="text-center p-4 bg-purple-50 rounded-lg">
            <div className="text-2xl font-bold text-purple-600 mb-1">≥4.2/5</div>
            <div className="text-sm text-gray-600">Voice Naturalness MOS</div>
          </div>
          
          <div className="text-center p-4 bg-orange-50 rounded-lg">
            <div className="text-2xl font-bold text-orange-600 mb-1">≥92%</div>
            <div className="text-sm text-gray-600">Lip-sync Accuracy</div>
          </div>
        </div>
      </motion.div>
    </div>
  )
}

export default SystemStatus