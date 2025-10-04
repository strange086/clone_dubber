import React, { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import axios from 'axios'

const API_BASE = '/api'

const VoiceProfileManager = () => {
  const [profiles, setProfiles] = useState([])
  const [loading, setLoading] = useState(true)
  const [showCreateForm, setShowCreateForm] = useState(false)
  const [createLoading, setCreateLoading] = useState(false)

  useEffect(() => {
    fetchProfiles()
  }, [])

  const fetchProfiles = async () => {
    try {
      const response = await axios.get(`${API_BASE}/voice-profiles`)
      setProfiles(response.data.profiles)
    } catch (error) {
      console.error('Error fetching profiles:', error)
    } finally {
      setLoading(false)
    }
  }

  const handleCreateProfile = async (formData) => {
    setCreateLoading(true)
    try {
      await axios.post(`${API_BASE}/voice-profiles/create`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      })
      setShowCreateForm(false)
      fetchProfiles()
    } catch (error) {
      console.error('Error creating profile:', error)
      alert('Error creating voice profile: ' + (error.response?.data?.detail || error.message))
    } finally {
      setCreateLoading(false)
    }
  }

  const handleDeleteProfile = async (profileId) => {
    if (!confirm('Are you sure you want to delete this voice profile?')) return
    
    try {
      await axios.delete(`${API_BASE}/voice-profiles/${profileId}`)
      fetchProfiles()
    } catch (error) {
      console.error('Error deleting profile:', error)
      alert('Error deleting profile: ' + (error.response?.data?.detail || error.message))
    }
  }

  const handleTestProfile = async (profileId, profileName) => {
    const text = prompt('Enter text to test the voice:', 'Hello, this is a test of my cloned voice.')
    if (!text) return

    try {
      const formData = new FormData()
      formData.append('text', text)
      formData.append('emotion', 'neutral')

      const response = await axios.post(`${API_BASE}/voice-profiles/${profileId}/test`, formData, {
        responseType: 'blob'
      })

      // Create audio element and play
      const audioUrl = URL.createObjectURL(response.data)
      const audio = new Audio(audioUrl)
      audio.play()
    } catch (error) {
      console.error('Error testing profile:', error)
      alert('Error testing profile: ' + (error.response?.data?.detail || error.message))
    }
  }

  if (loading) {
    return (
      <div className="flex justify-center items-center h-64">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
      </div>
    )
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex justify-between items-center">
        <div>
          <h2 className="text-2xl font-bold text-gray-900">Voice Profiles</h2>
          <p className="text-gray-600 mt-1">
            Manage voice profiles for high-quality voice cloning
          </p>
        </div>
        <button
          onClick={() => setShowCreateForm(true)}
          className="bg-blue-600 text-white px-4 py-2 rounded-lg hover:bg-blue-700 transition-colors"
        >
          + Create Profile
        </button>
      </div>

      {/* Create Profile Modal */}
      {showCreateForm && (
        <CreateProfileModal
          onClose={() => setShowCreateForm(false)}
          onSubmit={handleCreateProfile}
          loading={createLoading}
        />
      )}

      {/* Profiles Grid */}
      {profiles.length === 0 ? (
        <div className="text-center py-12">
          <div className="text-6xl mb-4">🎙️</div>
          <h3 className="text-lg font-medium text-gray-900 mb-2">No voice profiles yet</h3>
          <p className="text-gray-600 mb-4">
            Create your first voice profile to start cloning voices
          </p>
          <button
            onClick={() => setShowCreateForm(true)}
            className="bg-blue-600 text-white px-6 py-2 rounded-lg hover:bg-blue-700 transition-colors"
          >
            Create Your First Profile
          </button>
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {profiles.map((profile) => (
            <ProfileCard
              key={profile.profile_id}
              profile={profile}
              onDelete={() => handleDeleteProfile(profile.profile_id)}
              onTest={() => handleTestProfile(profile.profile_id, profile.speaker_name)}
            />
          ))}
        </div>
      )}
    </div>
  )
}

const CreateProfileModal = ({ onClose, onSubmit, loading }) => {
  const [formData, setFormData] = useState({
    speaker_name: '',
    target_language: 'hi',
    gender: 'male',
    audio_file: null
  })

  const languages = {
    'hi': 'Hindi',
    'bn': 'Bengali',
    'te': 'Telugu',
    'mr': 'Marathi',
    'ta': 'Tamil',
    'gu': 'Gujarati',
    'kn': 'Kannada',
    'ml': 'Malayalam',
    'pa': 'Punjabi',
    'ur': 'Urdu',
    'en': 'English'
  }

  const handleSubmit = (e) => {
    e.preventDefault()
    if (!formData.audio_file) {
      alert('Please select an audio file')
      return
    }

    const submitData = new FormData()
    submitData.append('speaker_name', formData.speaker_name)
    submitData.append('target_language', formData.target_language)
    submitData.append('gender', formData.gender)
    submitData.append('audio_file', formData.audio_file)

    onSubmit(submitData)
  }

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
      <motion.div
        initial={{ opacity: 0, scale: 0.95 }}
        animate={{ opacity: 1, scale: 1 }}
        className="bg-white rounded-lg p-6 w-full max-w-md mx-4"
      >
        <h3 className="text-lg font-medium text-gray-900 mb-4">Create Voice Profile</h3>
        
        <form onSubmit={handleSubmit} className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Speaker Name
            </label>
            <input
              type="text"
              required
              value={formData.speaker_name}
              onChange={(e) => setFormData({...formData, speaker_name: e.target.value})}
              className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
              placeholder="Enter speaker name"
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Target Language
            </label>
            <select
              value={formData.target_language}
              onChange={(e) => setFormData({...formData, target_language: e.target.value})}
              className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
            >
              {Object.entries(languages).map(([code, name]) => (
                <option key={code} value={code}>{name}</option>
              ))}
            </select>
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Gender
            </label>
            <select
              value={formData.gender}
              onChange={(e) => setFormData({...formData, gender: e.target.value})}
              className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
            >
              <option value="male">Male</option>
              <option value="female">Female</option>
            </select>
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Audio File (min 10 seconds)
            </label>
            <input
              type="file"
              required
              accept="audio/*"
              onChange={(e) => setFormData({...formData, audio_file: e.target.files[0]})}
              className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
            <p className="text-xs text-gray-500 mt-1">
              Upload clear audio with at least 10 seconds of speech
            </p>
          </div>

          <div className="flex justify-end space-x-3 pt-4">
            <button
              type="button"
              onClick={onClose}
              className="px-4 py-2 text-gray-700 border border-gray-300 rounded-md hover:bg-gray-50"
              disabled={loading}
            >
              Cancel
            </button>
            <button
              type="submit"
              disabled={loading}
              className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 disabled:opacity-50"
            >
              {loading ? 'Creating...' : 'Create Profile'}
            </button>
          </div>
        </form>
      </motion.div>
    </div>
  )
}

const ProfileCard = ({ profile, onDelete, onTest }) => {
  const getQualityColor = (score) => {
    if (score >= 0.8) return 'text-green-600 bg-green-100'
    if (score >= 0.6) return 'text-yellow-600 bg-yellow-100'
    return 'text-red-600 bg-red-100'
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="bg-white rounded-lg shadow-sm border p-6 hover:shadow-md transition-shadow"
    >
      <div className="flex items-start justify-between mb-4">
        <div>
          <h3 className="font-medium text-gray-900">{profile.speaker_name}</h3>
          <p className="text-sm text-gray-600">{profile.language} • {profile.gender}</p>
        </div>
        <span className={`px-2 py-1 text-xs font-medium rounded-full ${getQualityColor(profile.quality_score)}`}>
          {Math.round(profile.quality_score * 100)}% Quality
        </span>
      </div>

      <div className="space-y-2 mb-4">
        <div className="flex justify-between text-sm">
          <span className="text-gray-600">Duration:</span>
          <span className="font-medium">{Math.round(profile.duration)}s</span>
        </div>
        <div className="flex justify-between text-sm">
          <span className="text-gray-600">Created:</span>
          <span className="font-medium">
            {new Date(profile.created_at).toLocaleDateString()}
          </span>
        </div>
      </div>

      <div className="flex space-x-2">
        <button
          onClick={onTest}
          className="flex-1 bg-blue-50 text-blue-600 px-3 py-2 rounded-md text-sm font-medium hover:bg-blue-100 transition-colors"
        >
          🎵 Test Voice
        </button>
        <button
          onClick={onDelete}
          className="bg-red-50 text-red-600 px-3 py-2 rounded-md text-sm font-medium hover:bg-red-100 transition-colors"
        >
          🗑️
        </button>
      </div>
    </motion.div>
  )
}

export default VoiceProfileManager