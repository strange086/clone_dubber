import React, { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import axios from 'axios'

const API_BASE = '/api'

const DubbingInterface = () => {
  const [file, setFile] = useState(null)
  const [targetLanguage, setTargetLanguage] = useState('hi')
  const [useVoiceCloning, setUseVoiceCloning] = useState(true)
  const [enableLipsync, setEnableLipsync] = useState(true)
  const [uploading, setUploading] = useState(false)
  const [jobs, setJobs] = useState([])

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

  useEffect(() => {
    fetchJobs()
    const interval = setInterval(fetchJobs, 2000) // Poll every 2 seconds
    return () => clearInterval(interval)
  }, [])

  const fetchJobs = async () => {
    try {
      const response = await axios.get(`${API_BASE}/jobs`)
      setJobs(Object.entries(response.data.jobs).map(([id, job]) => ({ id, ...job })))
    } catch (error) {
      console.error('Error fetching jobs:', error)
    }
  }

  const handleUpload = async (e) => {
    e.preventDefault()
    if (!file) return

    setUploading(true)
    try {
      const formData = new FormData()
      formData.append('file', file)
      formData.append('target_language', targetLanguage)
      formData.append('source_language', 'auto')
      formData.append('preserve_emotion', 'true')
      formData.append('enable_lipsync', enableLipsync.toString())

      const uploadResponse = await axios.post(`${API_BASE}/upload`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      })

      const jobId = uploadResponse.data.job_id

      // Start processing
      await axios.post(`${API_BASE}/process/${jobId}`)

      setFile(null)
      fetchJobs()
    } catch (error) {
      console.error('Error uploading:', error)
      alert('Upload failed: ' + (error.response?.data?.detail || error.message))
    } finally {
      setUploading(false)
    }
  }

  const handleDownload = async (jobId, filename) => {
    try {
      const response = await axios.get(`${API_BASE}/download/${jobId}`, {
        responseType: 'blob'
      })
      
      const url = window.URL.createObjectURL(new Blob([response.data]))
      const link = document.createElement('a')
      link.href = url
      link.setAttribute('download', filename)
      document.body.appendChild(link)
      link.click()
      link.remove()
    } catch (error) {
      console.error('Error downloading:', error)
      alert('Download failed: ' + (error.response?.data?.detail || error.message))
    }
  }

  return (
    <div className="space-y-8">
      {/* Upload Section */}
      <div className="bg-white rounded-lg shadow-sm border p-6">
        <h2 className="text-xl font-semibold text-gray-900 mb-4">
          🎬 Video Dubbing with Voice Cloning
        </h2>
        
        <form onSubmit={handleUpload} className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Upload Video File
            </label>
            <div className="border-2 border-dashed border-gray-300 rounded-lg p-6 text-center hover:border-blue-400 transition-colors">
              <input
                type="file"
                accept="video/*"
                onChange={(e) => setFile(e.target.files[0])}
                className="hidden"
                id="video-upload"
              />
              <label htmlFor="video-upload" className="cursor-pointer">
                {file ? (
                  <div>
                    <div className="text-2xl mb-2">📹</div>
                    <p className="text-sm font-medium text-gray-900">{file.name}</p>
                    <p className="text-xs text-gray-500">{(file.size / 1024 / 1024).toFixed(2)} MB</p>
                  </div>
                ) : (
                  <div>
                    <div className="text-4xl mb-2">📁</div>
                    <p className="text-sm font-medium text-gray-900">Click to upload video</p>
                    <p className="text-xs text-gray-500">MP4, MOV, AVI, MKV up to 500MB</p>
                  </div>
                )}
              </label>
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Target Language
              </label>
              <select
                value={targetLanguage}
                onChange={(e) => setTargetLanguage(e.target.value)}
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
              >
                {Object.entries(languages).map(([code, name]) => (
                  <option key={code} value={code}>{name}</option>
                ))}
              </select>
            </div>

            <div className="space-y-3">
              <label className="flex items-center">
                <input
                  type="checkbox"
                  checked={useVoiceCloning}
                  onChange={(e) => setUseVoiceCloning(e.target.checked)}
                  className="rounded border-gray-300 text-blue-600 focus:ring-blue-500"
                />
                <span className="ml-2 text-sm text-gray-700">
                  🎙️ Use Voice Cloning (Sprint 5)
                </span>
              </label>

              <label className="flex items-center">
                <input
                  type="checkbox"
                  checked={enableLipsync}
                  onChange={(e) => setEnableLipsync(e.target.checked)}
                  className="rounded border-gray-300 text-blue-600 focus:ring-blue-500"
                />
                <span className="ml-2 text-sm text-gray-700">
                  👄 Enable Lip Sync
                </span>
              </label>
            </div>
          </div>

          <button
            type="submit"
            disabled={!file || uploading}
            className="w-full bg-blue-600 text-white py-3 px-4 rounded-lg font-medium hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
          >
            {uploading ? (
              <span className="flex items-center justify-center">
                <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white mr-2"></div>
                Processing...
              </span>
            ) : (
              '🚀 Start Dubbing Process'
            )}
          </button>
        </form>
      </div>

      {/* Jobs Section */}
      <div className="bg-white rounded-lg shadow-sm border p-6">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">
          Processing Queue
        </h3>

        {jobs.length === 0 ? (
          <div className="text-center py-8">
            <div className="text-4xl mb-2">⏳</div>
            <p className="text-gray-600">No jobs in queue</p>
          </div>
        ) : (
          <div className="space-y-4">
            {jobs.map((job) => (
              <JobCard
                key={job.id}
                job={job}
                onDownload={(filename) => handleDownload(job.id, filename)}
              />
            ))}
          </div>
        )}
      </div>
    </div>
  )
}

const JobCard = ({ job, onDownload }) => {
  const getStatusColor = (status) => {
    switch (status) {
      case 'completed': return 'bg-green-100 text-green-800'
      case 'failed': return 'bg-red-100 text-red-800'
      case 'processing': return 'bg-blue-100 text-blue-800'
      default: return 'bg-gray-100 text-gray-800'
    }
  }

  const getStatusIcon = (status) => {
    switch (status) {
      case 'completed': return '✅'
      case 'failed': return '❌'
      case 'processing': return '⚙️'
      default: return '⏳'
    }
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="border rounded-lg p-4 hover:shadow-sm transition-shadow"
    >
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center space-x-3">
          <span className="text-lg">{getStatusIcon(job.status)}</span>
          <div>
            <h4 className="font-medium text-gray-900">{job.original_filename}</h4>
            <p className="text-sm text-gray-600">
              {job.target_language} • {job.preserve_emotion ? 'Emotion preserved' : ''} • 
              {job.enable_lipsync ? ' Lip sync enabled' : ''}
            </p>
          </div>
        </div>
        <span className={`px-2 py-1 text-xs font-medium rounded-full ${getStatusColor(job.status)}`}>
          {job.status}
        </span>
      </div>

      {job.status === 'processing' && (
        <div className="mb-3">
          <div className="flex justify-between text-sm text-gray-600 mb-1">
            <span>{job.current_step || job.message}</span>
            <span>{Math.round(job.progress || 0)}%</span>
          </div>
          <div className="w-full bg-gray-200 rounded-full h-2">
            <div
              className="bg-blue-600 h-2 rounded-full transition-all duration-300"
              style={{ width: `${job.progress || 0}%` }}
            ></div>
          </div>
        </div>
      )}

      {job.status === 'completed' && (
        <div className="flex items-center justify-between">
          <div className="text-sm text-gray-600">
            {job.speakers_detected && (
              <span>👥 {job.speakers_detected.length} speakers detected</span>
            )}
            {job.voice_profiles_created && (
              <span className="ml-3">🎙️ {job.voice_profiles_created.length} voice profiles</span>
            )}
          </div>
          <button
            onClick={() => onDownload(`dubbed_${job.original_filename}`)}
            className="bg-green-50 text-green-600 px-3 py-1 rounded-md text-sm font-medium hover:bg-green-100 transition-colors"
          >
            📥 Download
          </button>
        </div>
      )}

      {job.status === 'failed' && (
        <div className="text-sm text-red-600">
          Error: {job.error || job.message}
        </div>
      )}
    </motion.div>
  )
}

export default DubbingInterface