import React, { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import axios from 'axios'
import VoiceProfileManager from './components/VoiceProfileManager'
import DubbingInterface from './components/DubbingInterface'
import SystemStatus from './components/SystemStatus'

const API_BASE = '/api'

function App() {
  const [activeTab, setActiveTab] = useState('dubbing')
  const [systemInfo, setSystemInfo] = useState(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    fetchSystemInfo()
  }, [])

  const fetchSystemInfo = async () => {
    try {
      const response = await axios.get(`${API_BASE}/system/info`)
      setSystemInfo(response.data)
    } catch (error) {
      console.error('Error fetching system info:', error)
    } finally {
      setLoading(false)
    }
  }

  const tabs = [
    { id: 'dubbing', name: 'Video Dubbing', icon: '🎬' },
    { id: 'voices', name: 'Voice Profiles', icon: '🎙️' },
    { id: 'system', name: 'System Status', icon: '⚙️' }
  ]

  if (loading) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto mb-4"></div>
          <p className="text-gray-600">Loading EduDub AI...</p>
        </div>
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100">
      {/* Header */}
      <header className="bg-white shadow-sm border-b">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center py-4">
            <div className="flex items-center">
              <h1 className="text-2xl font-bold text-gray-900">
                🎙️ EduDub AI
              </h1>
              <span className="ml-3 px-2 py-1 text-xs font-medium bg-blue-100 text-blue-800 rounded-full">
                Voice Cloning Platform
              </span>
            </div>
            <div className="flex items-center space-x-4">
              <span className="text-sm text-gray-500">
                {systemInfo?.voice_profiles_count || 0} Voice Profiles
              </span>
              <div className={`w-3 h-3 rounded-full ${
                systemInfo?.models?.voice_cloning_system ? 'bg-green-400' : 'bg-red-400'
              }`}></div>
            </div>
          </div>
        </div>
      </header>

      {/* Navigation Tabs */}
      <nav className="bg-white border-b">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex space-x-8">
            {tabs.map((tab) => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`py-4 px-1 border-b-2 font-medium text-sm transition-colors ${
                  activeTab === tab.id
                    ? 'border-blue-500 text-blue-600'
                    : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                }`}
              >
                <span className="mr-2">{tab.icon}</span>
                {tab.name}
              </button>
            ))}
          </div>
        </div>
      </nav>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <motion.div
          key={activeTab}
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.3 }}
        >
          {activeTab === 'dubbing' && <DubbingInterface />}
          {activeTab === 'voices' && <VoiceProfileManager />}
          {activeTab === 'system' && <SystemStatus systemInfo={systemInfo} />}
        </motion.div>
      </main>

      {/* Footer */}
      <footer className="bg-white border-t mt-12">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4">
          <div className="text-center text-sm text-gray-500">
            <p>EduDub AI - Indian Language Dubbing Platform with Voice Cloning</p>
            <p className="mt-1">Sprint 5: Voice Tracking & Artist Voice Preservation</p>
          </div>
        </div>
      </footer>
    </div>
  )
}

export default App